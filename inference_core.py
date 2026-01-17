import os
import sys
import cv2
import torch
import argparse
import numpy as np
import warnings
import torch.nn.functional as F

# Add repo root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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


# --- OPENCV FLOW (CPU) ---
def compute_flow_opencv(frames, downscale_factor=0.5):
    """
    Computes optical flow on CPU using OpenCV.
    Returns tensors on CPU.
    """
    print(f"🌊 Computing Flow (OpenCV)...")
    h, w = frames[0].shape[:2]
    h_s, w_s = int(h * downscale_factor), int(w * downscale_factor)

    gray_frames = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (w_s, h_s), interpolation=cv2.INTER_LINEAR)
        gray_frames.append(g)

    flows_forward = []
    flows_backward = []

    def calc_flow(prev, curr):
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Resize back
        flow_full = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        # Scale flow values
        flow_full *= (1.0 / downscale_factor)
        # To Tensor [1, 2, H, W]
        return torch.from_numpy(flow_full).permute(2, 0, 1).float().unsqueeze(0)

    for i in range(len(frames) - 1):
        flows_forward.append(calc_flow(gray_frames[i], gray_frames[i + 1]))

    for i in range(1, len(frames)):
        flows_backward.append(calc_flow(gray_frames[i], gray_frames[i - 1]))

    # Stack [1, T-1, 2, H, W]
    # Keep on CPU
    t_fwd = torch.stack(flows_forward, dim=1)
    t_bwd = torch.stack(flows_backward, dim=1)

    return t_fwd, t_bwd


def main(args):
    print("🚀 [Core] Starting ProPainter (Hybrid: CPU Flow -> GPU Inpaint)...")

    # Devices
    cpu = torch.device("cpu")
    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = os.path.dirname(os.path.abspath(__file__))
    fc_p = args.fc_model_path or os.path.join(base, 'weights', 'recurrent_flow_completion.pth')
    pp_p = args.model_path

    print("📦 Loading models...")
    # 1. Flow Completion -> CPU (Avoids CUDA Crashes)
    # Must be Float32
    fix_fc = RecurrentFlowCompleteNet(fc_p).to(cpu).float().eval()

    # 2. ProPainter -> GPU
    model = InpaintGenerator(model_path=pp_p).to(gpu).eval()

    # FP16 for GPU model
    use_half = False
    if torch.cuda.is_available():
        try:
            model = model.half()
            use_half = True
            print("✅ GPU FP16 Enabled")
        except:
            model = model.float()

    # Data
    f_files = sorted(
        [os.path.join(args.video, f) for f in os.listdir(args.video) if f.endswith(('.jpg', '.png', '.jpeg'))])
    m_files = sorted(
        [os.path.join(args.mask, f) for f in os.listdir(args.mask) if f.endswith(('.jpg', '.png', '.jpeg'))])
    if not f_files: raise ValueError("No frames")

    print(f"🔄 Processing {len(f_files)} frames...")

    # Load Raw (for OpenCV)
    raw_frames = [pad_img_to_modulo(imread(f), 16) for f in f_files]

    # Load Tensors (for GPU Model)
    video_list, mask_list = [], []
    for img, m_f in zip(raw_frames, m_files):
        # Image
        t_img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        video_list.append(t_img)

        # Mask
        m_img = pad_img_to_modulo((cv2.imread(m_f, 0) > 127).astype(np.uint8) * 255, 16)
        t_msk = (torch.from_numpy(m_img).float() / 255.0 > 0.5).float().unsqueeze(0)
        mask_list.append(t_msk)

    # 1. OPTICAL FLOW (CPU)
    # Returns CPU tensors
    flows_fwd, flows_bwd = compute_flow_opencv(raw_frames)

    # 2. FLOW COMPLETION (CPU)
    print("🌊 Completing Flows (on CPU)...")
    # Prepare masks for CPU [1, T, 1, H, W]
    masks_cpu = torch.stack(mask_list).unsqueeze(0).to(cpu).float()

    with torch.no_grad():
        # fix_fc runs on CPU, Float32. Safe from CUDA errors.
        updated_flows_cpu = fix_fc(flows_fwd, flows_bwd, masks_cpu)

    # 3. PROPAINTER INFERENCE (GPU)
    print("⚡ ProPainter Inference (on GPU)...")

    # Move everything to GPU
    frames_gpu = torch.stack(video_list).unsqueeze(0).to(gpu)
    masks_gpu = masks_cpu.to(gpu)

    # Flows come from CPU, move to GPU
    flow_fwd_gpu = updated_flows_cpu[0].to(gpu)
    flow_bwd_gpu = updated_flows_cpu[1].to(gpu)

    if use_half:
        frames_gpu = frames_gpu.half()
        masks_gpu = masks_gpu.half()
        flow_fwd_gpu = flow_fwd_gpu.half()
        flow_bwd_gpu = flow_bwd_gpu.half()

    with torch.no_grad():
        # Run main model
        pred = model(frames_gpu, (flow_fwd_gpu, flow_bwd_gpu), masks_gpu)[0]

    # Save
    print("💾 Saving...")
    os.makedirs(args.output, exist_ok=True)
    orig_h, orig_w = imread(f_files[0]).shape[:2]

    for i, f_path in enumerate(f_files):
        p = pred[i].permute(1, 2, 0).float().cpu().numpy().clip(0, 1)
        p = p[:orig_h, :orig_w, :]

        orig = imread(f_path).astype(float) / 255.0
        m = (cv2.imread(m_files[i], 0).astype(float) / 255.0 > 0.5)[:, :, None]
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
    parser.add_argument('--raft_model_path', type=str, default=None)
    parser.add_argument('--fc_model_path', type=str, default=None)
    parser.add_argument('--raft_iter', type=int, default=20)
    args = parser.parse_args()
    main(args)