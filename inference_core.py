import os
import sys
import cv2
import torch
import argparse
import numpy as np
import warnings
import torch.nn.functional as F

# --- PARANOID STABILITY SETTINGS ---
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# -----------------------------------

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.propainter import InpaintGenerator


def imread(img_path):
    img = cv2.imread(img_path)
    if img is None: raise ValueError(f"Failed to read: {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pad_img_to_modulo(img, mod):
    h, w = img.shape[:2]
    h_pad = ((h + mod - 1) // mod) * mod - h
    w_pad = ((w + mod - 1) // mod) * mod - w
    return cv2.copyMakeBorder(img, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)


def compute_flow_opencv(frames, downscale_factor=0.5):
    print(f"🌊 Computing Flow (OpenCV)...")
    h, w = frames[0].shape[:2]
    h_s, w_s = int(h * downscale_factor), int(w * downscale_factor)

    gray_frames = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (w_s, h_s), interpolation=cv2.INTER_LINEAR)
        gray_frames.append(g)

    flows_forward, flows_backward = [], []

    def calc_flow(prev, curr):
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_full = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        flow_full *= (1.0 / downscale_factor)
        # FORCE CONTIGUOUS FP32
        t = torch.from_numpy(flow_full).permute(2, 0, 1).float()
        return t.contiguous().unsqueeze(0)

    for i in range(len(frames) - 1):
        flows_forward.append(calc_flow(gray_frames[i], gray_frames[i + 1]))
    for i in range(1, len(frames)):
        flows_backward.append(calc_flow(gray_frames[i], gray_frames[i - 1]))

    return torch.stack(flows_forward, dim=1), torch.stack(flows_backward, dim=1)


def run_pipeline(args):
    print("🚀 [Core] Starting ProPainter (Paranoid Mode: FP32 Only)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("📦 Loading ProPainter (FP32)...")
    # STRICTLY FP32. Do NOT use .half()
    model = InpaintGenerator(model_path=args.model_path).to(device).float().eval()

    # 2. Load Data
    f_files = sorted(
        [os.path.join(args.video, f) for f in os.listdir(args.video) if f.endswith(('.jpg', '.png', '.jpeg'))])
    m_files = sorted(
        [os.path.join(args.mask, f) for f in os.listdir(args.mask) if f.endswith(('.jpg', '.png', '.jpeg'))])
    if not f_files: raise ValueError("No frames found")

    print(f"🔄 Processing {len(f_files)} frames...")

    raw_frames = [pad_img_to_modulo(imread(f), 16) for f in f_files]

    video_list, mask_list = [], []
    for img, m_f in zip(raw_frames, m_files):
        # Image
        t_img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        video_list.append(t_img.contiguous())

        # Mask
        m_img = pad_img_to_modulo((cv2.imread(m_f, 0) > 127).astype(np.uint8) * 255, 16)
        t_msk = (torch.from_numpy(m_img).float() / 255.0 > 0.5).float().unsqueeze(0)
        mask_list.append(t_msk.contiguous())

    # Stack and move to GPU
    # Ensure contiguous memory layout for everything sent to GPU
    masked_frames_gpu = torch.stack(video_list).unsqueeze(0).to(device).contiguous()
    masks_gpu = torch.stack(mask_list).unsqueeze(0).to(device).contiguous()

    # 3. Compute Flow (CPU)
    flows_fwd_cpu, flows_bwd_cpu = compute_flow_opencv(raw_frames)

    # Move flows to GPU (FP32)
    flows_fwd_gpu = flows_fwd_cpu.to(device).contiguous()
    flows_bwd_gpu = flows_bwd_cpu.to(device).contiguous()

    # 4. Inference
    print("⚡ ProPainter Inference (FP32)...")
    with torch.no_grad():
        # Passing tuple of flows
        pred = model(masked_frames_gpu, (flows_fwd_gpu, flows_bwd_gpu), masks_gpu)[0]

    # 5. Save
    print("💾 Saving...")
    os.makedirs(args.output, exist_ok=True)
    orig_h, orig_w = imread(f_files[0]).shape[:2]

    for i, f_path in enumerate(f_files):
        p = pred[i].permute(1, 2, 0).cpu().numpy().clip(0, 1)
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
    # Ignored args
    parser.add_argument('--raft_model_path', type=str, default=None)
    parser.add_argument('--fc_model_path', type=str, default=None)
    parser.add_argument('--raft_iter', type=int, default=20)
    args = parser.parse_args()

    try:
        run_pipeline(args)
    except Exception as e:
        print("\n❌ CRITICAL ERROR")
        import traceback

        traceback.print_exc()
        sys.exit(1)