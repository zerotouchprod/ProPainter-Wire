import os
import sys
import traceback

# --- DEBUG: FORCE SYNCHRONOUS CUDA EXECUTION ---
# Это заставит скрипт падать ровно на той строке, где ошибка, а не позже.
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# -----------------------------------------------

import cv2
import torch
import argparse
import numpy as np
import warnings
import torch.nn.functional as F

# Add repo root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.propainter import InpaintGenerator
from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet

warnings.filterwarnings("ignore")


# --- INLINE FIX: Safe Bilinear Sampler (No external deps) ---
def safe_bilinear_sampler(img, coords, mode='bilinear', mask=False):
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()
    return img


# ------------------------------------------------------------

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
    print(f"🚀 [Debug Mode] CUDA_LAUNCH_BLOCKING=1 enabled.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    base = os.path.dirname(os.path.abspath(__file__))
    raft_p = args.raft_model_path or os.path.join(base, 'weights', 'raft-things.pth')
    fc_p = args.fc_model_path or os.path.join(base, 'weights', 'recurrent_flow_completion.pth')
    pp_p = args.model_path

    print("📦 Loading models...")
    fix_raft = RAFT_bi(model_path=raft_p, device=device)
    fix_fc = RecurrentFlowCompleteNet(fc_p).to(device).eval()
    model = InpaintGenerator(model_path=pp_p).to(device).eval()

    # Data Loading
    frames = sorted(
        [os.path.join(args.video, f) for f in os.listdir(args.video) if f.endswith(('.jpg', '.png', '.jpeg'))])
    masks = sorted([os.path.join(args.mask, f) for f in os.listdir(args.mask) if f.endswith(('.jpg', '.png', '.jpeg'))])
    if not frames: raise ValueError("No frames found")

    print(f"🔄 Processing {len(frames)} frames...")

    # Load Batch
    video_data, mask_data = [], []
    for f, m in zip(frames, masks):
        img = pad_img_to_modulo(imread(f), 16)
        msk = pad_img_to_modulo((cv2.imread(m, 0) > 127).astype(np.uint8) * 255, 16)
        video_data.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
        mask_data.append((torch.from_numpy(msk).float() / 255.0 > 0.5).float().unsqueeze(0))

    masked_frames = torch.stack(video_data).unsqueeze(0).to(device)
    masks_tensor = torch.stack(mask_data).unsqueeze(0).to(device)

    # --- RAFT STEP ---
    print("🌊 Running RAFT...")
    # Downscale
    scale = 0.5
    b, t, c, h, w = masked_frames.shape
    h_s, w_s = int(h * scale), int(w * scale)

    vid_s = F.interpolate(masked_frames.view(-1, c, h, w), size=(h_s, w_s), mode='bilinear', align_corners=False)
    vid_s = vid_s.view(b, t, c, h_s, w_s)

    with torch.no_grad():
        # This is where it crashes usually.
        # Ensure input is contiguous
        vid_s = vid_s.contiguous()
        flows_s = fix_raft(vid_s, None)

    print("🌊 RAFT Completed. Upscaling flows...")
    flows_l = []
    for f in flows_s:
        up = F.interpolate(f.view(-1, 2, h_s, w_s), size=(h, w), mode='bilinear', align_corners=False)
        flows_l.append(up.view(b, t - 1, 2, h, w) * (1.0 / scale))
    gt_flows = tuple(flows_l)

    # --- INFERENCE STEP ---
    print("⚡ Running ProPainter Inference...")
    with torch.no_grad():
        updated_flows = fix_fc(gt_flows[0], gt_flows[1], masks_tensor)
        pred = model(masked_frames, updated_flows, masks_tensor)[0]

    print("💾 Saving...")
    os.makedirs(args.output, exist_ok=True)
    # Saving logic omitted for brevity in debug mode, but needed for job success
    for i, frame in enumerate(frames):
        # ... simple save ...
        p = pred[i].permute(1, 2, 0).cpu().numpy()
        cv2.imwrite(os.path.join(args.output, os.path.basename(frame)), (p * 255).astype(np.uint8))

    print("✅ Success.")


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

    try:
        main(args)
    except Exception:
        print("\n\n" + "=" * 60)
        print("🚨 CRITICAL FAILURE 🚨")
        print("Full Stack Trace (Synchronous Mode):")
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60 + "\n")
        sys.exit(1)