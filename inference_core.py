import os
import sys
import cv2
import torch
import argparse
import numpy as np
import warnings
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

# Add repo root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import ProPainter (Keep Flow Completion & Inpainting)
from model.propainter import InpaintGenerator
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


# --- OPENCV FLOW IMPLEMENTATION (ROBUST FALLBACK) ---
def compute_flow_opencv(frames, downscale_factor=0.5):
    """
    Computes bidirectional optical flow using OpenCV Farneback algorithm.
    Reliable, CPU-based, no CUDA crashes.
    """
    print(f"🌊 Computing Optical Flow (OpenCV Farneback)...")

    # 1. Prepare Frames (Gray + Downscale)
    h, w = frames[0].shape[:2]
    h_s, w_s = int(h * downscale_factor), int(w * downscale_factor)

    gray_frames = []
    for f in frames:
        # Convert torch [C,H,W] to numpy [H,W,C] -> Gray
        # But frames passed here are usually file paths in original script logic,
        # let's adapt to receive tensor or list of numpy.
        # Assuming frames is list of numpy arrays [H,W,3] RGB
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (w_s, h_s), interpolation=cv2.INTER_LINEAR)
        gray_frames.append(g)

    flows_forward = []
    flows_backward = []

    # Parallel processing for speed
    def calc_pair(i, direction):
        prev = gray_frames[i]
        curr = gray_frames[i + 1] if direction == 'fwd' else gray_frames[i - 1]

        # Farneback Params: pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Resize flow back to original resolution
        # Flow values scale with resolution!
        flow_full = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        flow_full *= (1.0 / downscale_factor)

        # Convert to Tensor [1, 2, H, W]
        t_flow = torch.from_numpy(flow_full).permute(2, 0, 1).float().unsqueeze(0)
        return t_flow

    # Compute Forward (0->1, 1->2...)
    for i in range(len(frames) - 1):
        flows_forward.append(calc_pair(i, 'fwd'))

    # Compute Backward (1->0, 2->1...)
    # Frame 0 has no backward flow from -1.
    # Logic: backward flow at index i corresponds to flow from i to i-1
    # RAFT usually returns N-1 flows.
    for i in range(1, len(frames)):
        flows_backward.append(calc_pair(i, 'bwd'))

    return flows_forward, flows_backward


# ----------------------------------------------------

def main(args):
    print("🚀 [Core] Starting ProPainter (OpenCV Flow Mode)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = os.path.dirname(os.path.abspath(__file__))
    fc_p = args.fc_model_path or os.path.join(base, 'weights', 'recurrent_flow_completion.pth')
    pp_p = args.model_path

    print("📦 Loading models (Skipping RAFT)...")
    fix_fc = RecurrentFlowCompleteNet(fc_p).to(device).eval()
    model = InpaintGenerator(model_path=pp_p).to(device).eval()

    # FP16
    use_half = False
    if torch.cuda.is_available():
        try:
            model = model.half()
            use_half = True
            print("✅ FP16 Enabled")
        except:
            model = model.float()

    # Data
    frame_files = sorted(
        [os.path.join(args.video, f) for f in os.listdir(args.video) if f.endswith(('.jpg', '.png', '.jpeg'))])
    mask_files = sorted(
        [os.path.join(args.mask, f) for f in os.listdir(args.mask) if f.endswith(('.jpg', '.png', '.jpeg'))])

    if not frame_files: raise ValueError("No frames found")

    print(f"🔄 Processing {len(frame_files)} frames...")

    # Load for Flow (Numpy)
    raw_frames = [pad_img_to_modulo(imread(f), 16) for f in frame_files]

    # Load for Model (Tensor)
    video_data = []
    mask_data = []
    for f, m in zip(frame_files, mask_files):
        img = pad_img_to_modulo(imread(f), 16)
        msk = pad_img_to_modulo((cv2.imread(m, 0) > 127).astype(np.uint8) * 255, 16)
        video_data.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
        mask_data.append((torch.from_numpy(msk).float() / 255.0 > 0.5).float().unsqueeze(0))

    masked_frames_gpu = torch.stack(video_data).unsqueeze(0).to(device)
    masks_gpu = torch.stack(mask_data).unsqueeze(0).to(device)

    # --- COMPUTE FLOW (OPENCV) ---
    # No RAFT model needed.
    flows_fwd_list, flows_bwd_list = compute_flow_opencv(raw_frames, downscale_factor=0.5)

    # Stack into tensors [1, T-1, 2, H, W]
    flows_fwd = torch.stack(flows_fwd_list, dim=1).to(device)
    flows_bwd = torch.stack(flows_bwd_list, dim=1).to(device)

    gt_flows = (flows_fwd, flows_bwd)
    torch.cuda.empty_cache()

    # --- INFERENCE ---
    print("⚡ ProPainter Inference...")
    with torch.no_grad():
        updated_flows = fix_fc(gt_flows[0], gt_flows[1], masks_gpu)

        in_v = masked_frames_gpu.half() if use_half else masked_frames_gpu
        in_m = masks_gpu.half() if use_half else masks_gpu
        in_f = (updated_flows[0].half(), updated_flows[1].half()) if use_half else updated_flows

        pred = model(in_v, in_f, in_m)[0]

    # Save
    print("💾 Saving...")
    os.makedirs(args.output, exist_ok=True)
    orig_h, orig_w = imread(frame_files[0]).shape[:2]

    for i, f_path in enumerate(frame_files):
        p = pred[i].permute(1, 2, 0)
        if use_half: p = p.float()
        p = p.cpu().numpy().clip(0, 1)
        p = p[:orig_h, :orig_w, :]

        orig = imread(f_path).astype(float) / 255.0
        m = (cv2.imread(mask_files[i], 0).astype(float) / 255.0 > 0.5)[:, :, None]
        final = p * m + orig * (1 - m)

        cv2.imwrite(os.path.join(args.output, os.path.basename(f_path)),
                    cv2.cvtColor((final * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    print("✅ Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--mask', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='weights/ProPainter.pth')
    # Compatibility args (ignored)
    parser.add_argument('--raft_model_path', type=str, default=None)
    parser.add_argument('--fc_model_path', type=str, default=None)
    parser.add_argument('--raft_iter', type=int, default=20)
    args = parser.parse_args()
    main(args)