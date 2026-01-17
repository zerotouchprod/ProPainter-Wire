import os
import cv2
import torch
import argparse
import numpy as np
import warnings
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.propainter import InpaintGenerator

# Suppress annoying warnings
warnings.filterwarnings("ignore")

def imread(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def pad_img_to_modulo(img, mod):
    """
    Pad image to be divisible by mod (usually 8 or 16).
    Fixes CUBLAS errors on modern GPUs.
    """
    h, w = img.shape[:2]
    h_pad = ((h + mod - 1) // mod) * mod - h
    w_pad = ((w + mod - 1) // mod) * mod - w
    # Use REFLECT padding to avoid sharp edges at borders
    return cv2.copyMakeBorder(img, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)

def main(args):
    print(f"🚀 [Core] Starting ProPainter Inference (Mixed Precision)...")
    print(f"   Video: {args.video}")
    print(f"   Mask:  {args.mask}")
    print(f"   Out:   {args.output}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load RAFT (Flow Estimation) - Keep in FP32 for stability
    print(f"📦 Loading RAFT from {args.raft_model_path}...")
    if not os.path.exists(args.raft_model_path):
        raise FileNotFoundError(f"RAFT weights not found at {args.raft_model_path}")
    fix_raft = RAFT_bi(args.raft_model_path, device)
    fix_raft.eval()
    
    # 2. Load Flow Completion Model - Keep in FP32
    print(f"📦 Loading Flow Completion from {args.fc_model_path}...")
    if not os.path.exists(args.fc_model_path):
        raise FileNotFoundError(f"Flow completion weights not found at {args.fc_model_path}")
    fix_flow_complete = RecurrentFlowCompleteNet(args.fc_model_path).to(device)
    fix_flow_complete.eval()
    
    # 3. Load ProPainter Model - Enable FP16 with safety patches
    print(f"📦 Loading ProPainter from {args.model_path}...")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"ProPainter weights not found at {args.model_path}")
    model = InpaintGenerator(model_path=args.model_path).to(device)
    model.eval()
    
    # Enable FP16 for ProPainter (memory efficient)
    use_half = False
    if torch.cuda.is_available():
        try:
            # Convert model to half precision
            model = model.half()
            use_half = True
            print("✅ Precision: FP16 Enabled for ProPainter (with Safe Attention Patches)")
        except Exception as e:
            print(f"⚠️ FP16 failed, falling back to FP32. Error: {e}")
            model = model.float()
            use_half = False
    else:
        print("⚠️ Running on CPU (Slow!)")
        model = model.float()
    
    # 4. Prepare Data
    frames = sorted([os.path.join(args.video, f) for f in os.listdir(args.video) if f.endswith(('.jpg', '.png', '.jpeg'))])
    masks = sorted([os.path.join(args.mask, f) for f in os.listdir(args.mask) if f.endswith(('.jpg', '.png', '.jpeg'))])

    if len(frames) == 0:
        raise ValueError(f"No frames found in {args.video}")
    if len(frames) != len(masks):
        raise ValueError(f"Mismatch: {len(frames)} frames vs {len(masks)} masks")

    # Read reference for size
    ref_img = imread(frames[0])
    orig_h, orig_w = ref_img.shape[:2]
    print(f"📏 Resolution: {orig_w}x{orig_h}")

    os.makedirs(args.output, exist_ok=True)

    # 5. Processing Loop
    video_data = []
    mask_data = []

    print("🔄 Pre-processing frames...")
    for f_path, m_path in zip(frames, masks):
        img = imread(f_path)
        msk = cv2.imread(m_path, 0)
        
        # Force strict binary mask (0 or 255) to avoid gray artifacts
        msk = (msk > 127).astype(np.uint8) * 255

        # Resize/Pad logic (CRITICAL for CUDA stability)
        img_padded = pad_img_to_modulo(img, 16)
        msk_padded = pad_img_to_modulo(msk, 16)

        img_t = torch.from_numpy(img_padded).permute(2, 0, 1).float() / 255.0
        msk_t = torch.from_numpy(msk_padded).float() / 255.0
        
        # Mask needs specific shape: [1, H, W]
        msk_t = (msk_t > 0.5).float().unsqueeze(0)

        video_data.append(img_t)
        mask_data.append(msk_t)

    # Stack to Tensor: [1, T, C, H, W]
    video_tensor = torch.stack(video_data).unsqueeze(0).to(device)
    mask_tensor = torch.stack(mask_data).unsqueeze(0).to(device)

    # 6. Compute Flows (FP32 for stability)
    print("🌊 Computing optical flows with RAFT (FP32)...")
    with torch.no_grad():
        # RAFT expects frames in FP32
        gt_flows_bi = fix_raft(video_tensor.float(), iters=args.raft_iter)
    
    # 7. Complete Flows (FP32)
    print("🔄 Completing flows...")
    with torch.no_grad():
        pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, mask_tensor.float())
        pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, mask_tensor.float())
    
    # 8. Temporal Propagation
    print("🌀 Temporal propagation...")
    with torch.no_grad():
        prop_imgs, updated_local_masks = model.img_propagation(
            video_tensor * (1 - mask_tensor),  # masked frames
            pred_flows_bi, 
            mask_tensor, 
            'nearest'
        )
    
    b, t, _, h, w = mask_tensor.shape
    updated_masks = updated_local_masks.view(b, t, 1, h, w)
    updated_frames = video_tensor * (1 - mask_tensor) + prop_imgs.view(b, t, 3, h, w) * mask_tensor
    
    # 9. Prepare inputs for final inference
    # Cast to half precision if model is half
    if use_half:
        video_tensor = video_tensor.half()
        mask_tensor = mask_tensor.half()
        updated_frames = updated_frames.half()
        updated_masks = updated_masks.half()
        # Cast flows to half
        pred_flows_bi = (pred_flows_bi[0].half(), pred_flows_bi[1].half())
    
    # 10. Final Inference (Local window processing)
    print("⚡ Running final inference (ProPainter)...")
    
    # Simple approach: process all frames as one local window
    # For better quality, implement sliding window like in evaluation script
    # But for simplicity, we'll process all frames together
    neighbor_length = min(t, args.neighbor_length)
    neighbor_stride = neighbor_length // 2
    
    comp_frames = [None] * t
    
    for f in range(0, t, neighbor_stride):
        neighbor_ids = [
            i for i in range(max(0, f - neighbor_stride),
                             min(t, f + neighbor_stride + 1))
        ]
        # Use all frames as reference (simplified)
        ref_ids = []
        for i in range(0, t, args.ref_stride):
            if i not in neighbor_ids:
                ref_ids.append(i)
        
        selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
        selected_masks = mask_tensor[:, neighbor_ids + ref_ids, :, :, :]
        selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
        selected_pred_flows_bi = (
            pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :] if len(neighbor_ids) > 1 else pred_flows_bi[0][:, :0, :, :, :],
            pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :] if len(neighbor_ids) > 1 else pred_flows_bi[1][:, :0, :, :, :]
        )
        
        with torch.no_grad():
            l_t = len(neighbor_ids)
            pred_img = model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
            pred_img = pred_img.view(-1, 3, h, w)
            
            # Convert to numpy
            pred_img = (pred_img + 1) / 2  # from [-1,1] to [0,1]
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
            
            binary_masks = mask_tensor[0, neighbor_ids, :, :, :].cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
            
            for i, idx in enumerate(neighbor_ids):
                img_np = pred_img[i].astype(np.uint8) * binary_masks[i] + \
                         (video_tensor[0, idx].cpu().permute(1, 2, 0).numpy() * 255 * (1 - binary_masks[i])).astype(np.uint8)
                
                if comp_frames[idx] is None:
                    comp_frames[idx] = img_np
                else:
                    comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img_np.astype(np.float32) * 0.5
    
    # 11. Save results
    print("💾 Saving results...")
    for i, frame in enumerate(comp_frames):
        if frame is None:
            # Fallback to original frame if something went wrong
            frame = (video_tensor[0, i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Background preservation (optional)
        orig_img = imread(frames[i]).astype(np.float32) / 255.0
        orig_mask = cv2.imread(masks[i], 0).astype(np.float32) / 255.0
        orig_mask = (orig_mask > 0.5)[:, :, None]
        
        frame_float = frame.astype(np.float32) / 255.0
        final_img = frame_float * orig_mask + orig_img * (1 - orig_mask)
        
        save_path = os.path.join(args.output, os.path.basename(frames[i]))
        final_bgr = cv2.cvtColor((np.clip(final_img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, final_bgr)
    
    print("✅ Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='Path to input frames folder')
    parser.add_argument('--mask', type=str, required=True, help='Path to input masks folder')
    parser.add_argument('--output', type=str, required=True, help='Path to output folder')
    parser.add_argument('--model_path', type=str, default='weights/ProPainter.pth', help='Path to ProPainter .pth model')
    parser.add_argument('--raft_model_path', type=str, default='weights/raft-things.pth', help='Path to RAFT .pth model')
    parser.add_argument('--fc_model_path', type=str, default='weights/recurrent_flow_completion.pth', help='Path to flow completion .pth model')
    parser.add_argument('--raft_iter', type=int, default=20, help='RAFT iterations')
    parser.add_argument('--ref_stride', type=int, default=10, help='Reference frame stride')
    parser.add_argument('--neighbor_length', type=int, default=20, help='Neighbor window length')
    args = parser.parse_args()
    main(args)
