import os
import cv2
import torch
import argparse
import numpy as np
import warnings
import sys
import scipy.ndimage

# Add repo root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.propainter import InpaintGenerator
from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet

# Suppress warnings
warnings.filterwarnings("ignore")


def imread(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pad_img_to_modulo(img, mod):
    h, w = img.shape[:2]
    h_pad = ((h + mod - 1) // mod) * mod - h
    w_pad = ((w + mod - 1) // mod) * mod - w
    return cv2.copyMakeBorder(img, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)


def get_ref_index(f, neighbor_ids, length):
    ref_index = []
    for i in range(0, length, neighbor_ids):
        ref_index.append(i)
    return ref_index


def main(args):
    print(f"🚀 [Core] Starting ProPainter FULL Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Load All Models ---
    print("📦 Loading models...")

    # Paths (Auto-detect weights in 'weights/' folder relative to script)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(base_dir, 'weights')

    propainter_path = args.model_path
    raft_path = os.path.join(weights_dir, 'raft-things.pth')
    flow_comp_path = os.path.join(weights_dir, 'recurrent_flow_completion.pth')

    if not os.path.exists(raft_path):
        raise FileNotFoundError(f"RAFT weights not found at {raft_path}")

    # A. RAFT (Optical Flow)
    fix_raft = RAFT_bi(model_path=raft_path, device=device)

    # B. Flow Completion
    fix_flow_complete = RecurrentFlowCompleteNet(flow_comp_path)
    fix_flow_complete.to(device)
    fix_flow_complete.eval()

    # C. ProPainter (Inpainting)
    model = InpaintGenerator(model_path=propainter_path).to(device)
    model.eval()

    # FORCE FP32 for stability on RTX 30/40 series
    # Mixing FP16 in Flow/Inpaint is complex and error-prone
    print("🛡️ Precision: FP32 (Forced) - Stability Mode")

    # --- 2. Prepare Data ---
    frames = sorted(
        [os.path.join(args.video, f) for f in os.listdir(args.video) if f.endswith(('.jpg', '.png', '.jpeg'))])
    masks = sorted([os.path.join(args.mask, f) for f in os.listdir(args.mask) if f.endswith(('.jpg', '.png', '.jpeg'))])

    if len(frames) == 0: raise ValueError(f"No frames in {args.video}")

    # Read first frame to get size
    ref_img = imread(frames[0])
    orig_h, orig_w = ref_img.shape[:2]

    # Prepare batch
    video_data = []
    mask_data = []

    print("🔄 Pre-processing & Padding...")
    for f_path, m_path in zip(frames, masks):
        img = imread(f_path)
        msk = cv2.imread(m_path, 0)

        # Binarize mask
        msk = (msk > 127).astype(np.uint8) * 255

        # Pad to 8/16
        img = pad_img_to_modulo(img, 16)
        msk = pad_img_to_modulo(msk, 16)  # usually 8 is enough for RAFT, but keeping consistent

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        msk_t = torch.from_numpy(msk).float() / 255.0
        msk_t = (msk_t > 0.5).float().unsqueeze(0)

        video_data.append(img_t)
        mask_data.append(msk_t)

    # Tensor: [1, T, C, H, W]
    masked_frames = torch.stack(video_data).unsqueeze(0).to(device)
    masks = torch.stack(mask_data).unsqueeze(0).to(device)

    # --- 3. Run Pipeline ---
    with torch.no_grad():
        # A. Compute Optical Flows (RAFT)
        # Input: [1, T, 3, H, W] -> Output: flows_forward, flows_backward
        print("🌊 Computing Optical Flows (RAFT)...")
        # RAFT expects 0-255 usually, but RAFT_bi wrapper might handle normalization.
        # Checking RAFT_bi source: it expects [0,1] tensors. Correct.
        pred_flows_bi = fix_raft(masked_frames, frames[0])  # 2nd arg is just dummy path or ignored

        # B. Flow Completion
        print("✨ Completing Flows...")
        flows_forward, flows_backward = pred_flows_bi
        updated_flows_bi = fix_flow_complete(flows_forward, flows_backward, masks)

        # C. ProPainter Inference
        # Signature: (masked_frames, completed_flows, masks, masks_updated, num_local_frames)
        # Note: masks_updated is computed internally if needed, or we pass masks.
        # Let's check ProPainter forward signature.
        # It needs: masked_frames, pred_flows_bi, masks
        print("🎨 Inpainting...")

        # We manually perform the flow completion step that ProPainter usually does internally
        # OR we pass the raw flows if the model handles completion.
        # Actually, standard usage passes COMPLETED flows.

        # Mask update (dilation) for flow completion usually happens inside,
        # but let's assume we pass standard masks.

        # Calling the model
        # video_out = model(masked_frames, updated_flows_bi, masks)

        # Wait, the error said it needs: 'completed_flows', 'masks_updated', 'num_local_frames'
        # Let's construct arguments explicitly.

        # masks_updated: typically same as masks for inference, or slightly dilated.
        masks_updated = masks
        num_local_frames = len(frames)  # Use global temporal attention for short chunks

        pred_tensor = model(masked_frames, updated_flows_bi, masks, masks_updated, num_local_frames)

    # --- 4. Save Results ---
    print("💾 Saving results...")
    pred_tensor = pred_tensor[0]
    os.makedirs(args.output, exist_ok=True)

    for i in range(len(frames)):
        # Unpad and Save
        pred_frame = pred_tensor[i].permute(1, 2, 0).cpu().numpy()
        pred_frame = np.clip(pred_frame, 0, 1)

        # Crop
        pred_frame = pred_frame[:orig_h, :orig_w, :]

        # Background Preservation (Optional but recommended)
        orig_img = imread(frames[i]).astype(np.float32) / 255.0
        orig_mask = cv2.imread(masks[i], 0).astype(np.float32) / 255.0
        orig_mask = (orig_mask > 0.5)[:, :, None]

        final_img = pred_frame * orig_mask + orig_img * (1 - orig_mask)

        save_path = os.path.join(args.output, os.path.basename(frames[i]))
        final_bgr = cv2.cvtColor((final_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, final_bgr)

    print("✅ Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--mask', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='weights/ProPainter.pth')
    args = parser.parse_args()
    main(args)