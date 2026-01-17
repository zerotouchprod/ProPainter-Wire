import os
import sys
import cv2
import torch
import argparse
import numpy as np
import warnings
import torch.nn.functional as F

# Disable Warnings
warnings.filterwarnings("ignore")

# Add Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Models
from model.propainter import InpaintGenerator
from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet


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
    print("🚀 [Core] Starting ProPainter (CPU-Safe Mode)...")

    # Force RAFT to CPU to avoid ANY CUDA kernel issues
    raft_device = torch.device("cpu")
    gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = os.path.dirname(os.path.abspath(__file__))
    raft_p = args.raft_model_path or os.path.join(base, 'weights', 'raft-things.pth')
    fc_p = args.fc_model_path or os.path.join(base, 'weights', 'recurrent_flow_completion.pth')
    pp_p = args.model_path

    print("📦 Loading models...")
    # RAFT on CPU
    fix_raft = RAFT_bi(model_path=raft_p, device=raft_device)
    # Others on GPU
    fix_fc = RecurrentFlowCompleteNet(fc_p).to(gpu_device).eval()
    model = InpaintGenerator(model_path=pp_p).to(gpu_device).eval()

    use_half = False
    if torch.cuda.is_available():
        try:
            model = model.half()
            use_half = True
            print("✅ FP16 Enabled (Model)")
        except:
            model = model.float()

    frames = sorted(
        [os.path.join(args.video, f) for f in os.listdir(args.video) if f.endswith(('.jpg', '.png', '.jpeg'))])
    masks = sorted([os.path.join(args.mask, f) for f in os.listdir(args.mask) if f.endswith(('.jpg', '.png', '.jpeg'))])
    if not frames: raise ValueError("No frames")

    print(f"🔄 Processing {len(frames)} frames...")
    video_data, mask_data = [], []
    for f, m in zip(frames, masks):
        img = pad_img_to_modulo(imread(f), 16)
        msk = pad_img_to_modulo((cv2.imread(m, 0) > 127).astype(np.uint8) * 255, 16)
        video_data.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
        mask_data.append((torch.from_numpy(msk).float() / 255.0 > 0.5).float().unsqueeze(0))

    # Tensors
    masked_frames_gpu = torch.stack(video_data).unsqueeze(0).to(gpu_device)
    masks_gpu = torch.stack(mask_data).unsqueeze(0).to(gpu_device)

    # --- RAFT (CPU Execution) ---
    print("🌊 Running RAFT (on CPU)...")
    scale = 0.5
    b, t, c, h, w = masked_frames_gpu.shape
    h_s, w_s = int(h * scale), int(w * scale)

    # Downscale on GPU, move to CPU
    vid_s_gpu = F.interpolate(masked_frames_gpu.view(-1, c, h, w), size=(h_s, w_s), mode='bilinear',
                              align_corners=False)
    vid_s_cpu = vid_s_gpu.view(b, t, c, h_s, w_s).cpu()

    with torch.no_grad():
        flows_s_cpu = fix_raft(vid_s_cpu, None)

    # Move to GPU & Upscale
    print("🌊 RAFT Done. Moving to GPU...")
    flows_l = []
    for f in flows_s_cpu:
        f_gpu = f.to(gpu_device)
        up = F.interpolate(f_gpu.view(-1, 2, h_s, w_s), size=(h, w), mode='bilinear', align_corners=False)
        flows_l.append(up.view(b, t - 1, 2, h, w) * (1.0 / scale))
    gt_flows = tuple(flows_l)

    del vid_s_cpu, flows_s_cpu
    torch.cuda.empty_cache()

    # --- INFERENCE ---
    print("⚡ Inference...")
    with torch.no_grad():
        updated_flows = fix_fc(gt_flows[0], gt_flows[1], masks_gpu)
        in_v = masked_frames_gpu.half() if use_half else masked_frames_gpu
        in_m = masks_gpu.half() if use_half else masks_gpu
        in_f = (updated_flows[0].half(), updated_flows[1].half()) if use_half else updated_flows
        pred = model(in_v, in_f, in_m)[0]

    # Save
    print("💾 Saving...")
    os.makedirs(args.output, exist_ok=True)
    orig_h, orig_w = imread(frames[0]).shape[:2]

    for i, frame in enumerate(frames):
        p = pred[i].permute(1, 2, 0)
        if use_half: p = p.float()
        p = p.cpu().numpy().clip(0, 1)
        p = p[:orig_h, :orig_w, :]

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