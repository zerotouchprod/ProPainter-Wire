import sys
import types
import warnings

# --- NUCLEAR MONKEY PATCH: DISABLE BROKEN CUDA KERNELS ---
# This must run BEFORE any model imports!
# We create a dummy module that does nothing, preventing the real 'alt_cuda_corr' from loading.
dummy_cuda = types.ModuleType('alt_cuda_corr')
dummy_cuda.CorrelationFunction = None
sys.modules['alt_cuda_corr'] = dummy_cuda
sys.modules['model.modules.RAFT.core.alt_cuda_corr'] = dummy_cuda
print("🛡️ [Core] 'alt_cuda_corr' has been successfully neutralized via Monkey Patch.")
# ---------------------------------------------------------

import os
import cv2
import torch
import argparse
import numpy as np
import torch.nn.functional as F

# Add repo root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.propainter import InpaintGenerator
from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet

warnings.filterwarnings("ignore")


def imread(img_path):
    img = cv2.imread(img_path)
    if img is None: raise ValueError(f"Failed to read: {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pad_img_to_modulo(img, mod):
    h, w = img.shape[:2]
    h_pad = ((h + mod - 1) // mod) * mod - h
    w_pad = ((w + mod - 1) // mod) * mod - w
    return cv2.copyMakeBorder(img, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)


def main(args):
    print(f"🚀 [Core] Starting ProPainter (Nuclear Safe Mode)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    base = os.path.dirname(os.path.abspath(__file__))
    raft_p = args.raft_model_path or os.path.join(base, 'weights', 'raft-things.pth')
    fc_p = args.fc_model_path or os.path.join(base, 'weights', 'recurrent_flow_completion.pth')
    pp_p = args.model_path

    # Load Models
    print("📦 Loading models...")
    fix_raft = RAFT_bi(model_path=raft_p, device=device)
    fix_fc = RecurrentFlowCompleteNet(fc_p).to(device).eval()
    model = InpaintGenerator(model_path=pp_p).to(device).eval()

    # FP16 Logic
    use_half = False
    if torch.cuda.is_available():
        try:
            model = model.half()
            use_half = True
            print("✅ ProPainter: FP16 Enabled")
        except:
            model = model.float()

    # Data
    frames = sorted(
        [os.path.join(args.video, f) for f in os.listdir(args.video) if f.endswith(('.jpg', '.png', '.jpeg'))])
    masks = sorted([os.path.join(args.mask, f) for f in os.listdir(args.mask) if f.endswith(('.jpg', '.png', '.jpeg'))])
    if not frames: raise ValueError("No frames found")

    # Pre-process
    print("🔄 Pre-processing...")
    video_data, mask_data = [], []
    h_orig, w_orig = imread(frames[0]).shape[:2]

    for f, m in zip(frames, masks):
        img = pad_img_to_modulo(imread(f), 16)
        msk = pad_img_to_modulo((cv2.imread(m, 0) > 127).astype(np.uint8) * 255, 16)
        video_data.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
        mask_data.append((torch.from_numpy(msk).float() / 255.0 > 0.5).float().unsqueeze(0))

    masked_frames = torch.stack(video_data).unsqueeze(0).to(device)
    masks_tensor = torch.stack(mask_data).unsqueeze(0).to(device)

    # RAFT (Downscaled & Safe)
    print("🌊 RAFT (Smart Downscale)...")
    scale = 0.5
    b, t, c, h, w = masked_frames.shape
    h_s, w_s = int(h * scale), int(w * scale)

    # Downscale
    vid_s = F.interpolate(masked_frames.view(-1, c, h, w), size=(h_s, w_s), mode='bilinear', align_corners=False)
    vid_s = vid_s.view(b, t, c, h_s, w_s)

    with torch.no_grad():
        # RAFT runs in FP32. Because we monkey-patched alt_cuda_corr,
        # it will fail if the fallback isn't triggered or if logic depends on it.
        # But standard RAFT implementation usually has a try-except block.
        # Our patch makes the import succeed but return None, forcing the fallback path in properly written code.
        flows_s = fix_raft(vid_s, None)

    # Upscale
    flows_l = []
    for f in flows_s:
        up = F.interpolate(f.view(-1, 2, h_s, w_s), size=(h, w), mode='bilinear', align_corners=False)
        flows_l.append(up.view(b, t - 1, 2, h, w) * (1.0 / scale))

    gt_flows = tuple(flows_l)
    del vid_s, flows_s
    torch.cuda.empty_cache()

    # Inference
    print("⚡ ProPainter Inference...")
    with torch.no_grad():
        updated_flows = fix_fc(gt_flows[0], gt_flows[1], masks_tensor)

        in_v = masked_frames.half() if use_half else masked_frames
        in_m = masks_tensor.half() if use_half else masks_tensor
        in_f = (updated_flows[0].half(), updated_flows[1].half()) if use_half else updated_flows

        pred = model(in_v, in_f, in_m)[0]

    # Save
    print("💾 Saving...")
    os.makedirs(args.output, exist_ok=True)
    for i, frame in enumerate(frames):
        p = pred[i].permute(1, 2, 0)
        if use_half: p = p.float()
        p = p.cpu().numpy().clip(0, 1)
        p = p[:h_orig, :w_orig, :]  # Crop padding

        orig = imread(frame).astype(float) / 255.0
        m = (cv2.imread(masks[i], 0).astype(float) / 255.0 > 0.5)[:, :, None]
        final = p * m + orig * (1 - m)

        cv2.imwrite(os.path.join(args.output, os.path.basename(frame)),
                    cv2.cvtColor((final * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    print("✅ Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--mask', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='weights/ProPainter.pth')
    parser.add_argument('--raft_model_path', type=str, default=None)
    parser.add_argument('--fc_model_path', type=str, default=None)
    parser.add_argument('--raft_iter', type=int, default=20)
    args = parser.parse_args()
    main(args)