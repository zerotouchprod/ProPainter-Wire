import os
import cv2
import torch
import argparse
import numpy as np
import warnings
import sys
import torch.nn.functional as F

# Add repo root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.propainter import InpaintGenerator
from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet

# Disable custom CUDA kernels if they exist (prevents CUDA errors)
try:
    import model.modules.RAFT.core.corr
    model.modules.RAFT.core.corr.alt_cuda_corr = None
except:
    pass

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

def main(args):
    print(f"🚀 [Core] Starting ProPainter (Stable Version)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(base_dir, 'weights')
    
    # Paths
    raft_path = args.raft_model_path if args.raft_model_path else os.path.join(weights_dir, 'raft-things.pth')
    fc_path = args.fc_model_path if args.fc_model_path else os.path.join(weights_dir, 'recurrent_flow_completion.pth')
    propainter_path = args.model_path

    print("📦 Loading models...")
    # 1. RAFT (FP32)
    fix_raft = RAFT_bi(model_path=raft_path, device=device)
    
    # 2. Flow Completion (FP32)
    fix_flow_complete = RecurrentFlowCompleteNet(fc_path)
    fix_flow_complete.to(device).eval()
    
    # 3. ProPainter (FP16/FP32 Mixed)
    model = InpaintGenerator(model_path=propainter_path).to(device).eval()

    # Smart Precision Setup
    use_half = False
    if torch.cuda.is_available():
        try:
            # We use half precision for the main model to save VRAM
            model = model.half()
            use_half = True
            print("✅ Precision: FP16 Enabled")
        except:
            print("⚠️ FP16 failed, using FP32")
            model = model.float()

    # Data Loading
    frames = sorted([os.path.join(args.video, f) for f in os.listdir(args.video) if f.endswith(('.jpg', '.png', '.jpeg'))])
    masks = sorted([os.path.join(args.mask, f) for f in os.listdir(args.mask) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    if not frames: raise ValueError("No frames found")
    
    # Pre-processing
    print("🔄 Pre-processing...")
    video_data = []
    mask_data = []
    
    # Read first frame for dimensions
    ref = imread(frames[0])
    h_orig, w_orig = ref.shape[:2]

    for f_path, m_path in zip(frames, masks):
        img = pad_img_to_modulo(imread(f_path), 16)
        msk = pad_img_to_modulo((cv2.imread(m_path, 0) > 127).astype(np.uint8) * 255, 16)
        
        video_data.append(torch.from_numpy(img).permute(2,0,1).float()/255.0)
        mask_data.append((torch.from_numpy(msk).float()/255.0 > 0.5).float().unsqueeze(0))

    masked_frames = torch.stack(video_data).unsqueeze(0).to(device)
    masks_tensor = torch.stack(mask_data).unsqueeze(0).to(device)

    # --- MEMORY EFFICIENT RAFT ---
    print("🌊 RAFT (Smart Downscale)...")
    # Downscale by 0.5 to save ~75% VRAM during flow calculation
    scale_factor = 0.5
    b, t, c, h, w = masked_frames.shape
    h_small, w_small = int(h * scale_factor), int(w * scale_factor)
    
    # Resize for RAFT
    video_flat = masked_frames.view(-1, c, h, w)
    video_small = F.interpolate(video_flat, size=(h_small, w_small), mode='bilinear', align_corners=False)
    video_small = video_small.view(b, t, c, h_small, w_small)
    
    with torch.no_grad():
        # Run RAFT (FP32)
        flows_small = fix_raft(video_small, None)
    
    # Upscale Flows
    flows_large = []
    for flow in flows_small:
        # flow: [B, T-1, 2, H_small, W_small]
        bf, tf, cf, hf, wf = flow.shape
        flow_flat = flow.view(-1, cf, hf, wf)
        
        # Interpolate back
        upscaled = F.interpolate(flow_flat, size=(h, w), mode='bilinear', align_corners=False)
        
        # Scale values
        upscaled = upscaled * (1.0 / scale_factor)
        
        flows_large.append(upscaled.view(bf, tf, cf, h, w))
    
    gt_flows = tuple(flows_large)
    
    # Cleanup
    del video_small, flows_small, video_flat
    torch.cuda.empty_cache()
    
    # --- INFERENCE ---
    print("⚡ Running ProPainter...")
    with torch.no_grad():
        # Flow Completion (FP32)
        updated_flows = fix_flow_complete(gt_flows[0], gt_flows[1], masks_tensor)
        
        # Cast to FP16 for model if enabled
        inputs = masked_frames.half() if use_half else masked_frames
        masks_in = masks_tensor.half() if use_half else masks_tensor
        flows_in = (updated_flows[0].half(), updated_flows[1].half()) if use_half else updated_flows
        
        # Run Model
        pred = model(inputs, flows_in, masks_in)[0]

    # --- SAVE ---
    print("💾 Saving...")
    os.makedirs(args.output, exist_ok=True)
    for i, frame in enumerate(frames):
        p = pred[i].permute(1,2,0)
        if use_half: p = p.float()
        p = p.cpu().numpy().clip(0,1)
        
        # Crop padding
        p = p[:h_orig, :w_orig, :]
        
        # Background preservation
        orig = imread(frame).astype(float)/255.0
        m = (cv2.imread(masks[i], 0).astype(float)/255.0 > 0.5)[:,:,None]
        
        final = p * m + orig * (1 - m)
        
        save_path = os.path.join(args.output, os.path.basename(frame))
        cv2.imwrite(save_path, cv2.cvtColor((final*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    print("✅ Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--mask', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='weights/ProPainter.pth')
    # Optional args for compatibility
    parser.add_argument('--raft_model_path', type=str, default=None)
    parser.add_argument('--fc_model_path', type=str, default=None)
    parser.add_argument('--raft_iter', type=int, default=20)
    args = parser.parse_args()
    main(args)
