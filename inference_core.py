import os
import sys
import cv2
import torch
import argparse
import numpy as np
import warnings
import re
from PIL import Image

# 1. TUNING FOR STABILITY
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.propainter import InpaintGenerator


def smart_imread(img_path, grayscale=False):
    """
    Robust Reader. Handles:
    1. Broken Symlinks
    2. Missing Files
    3. Index Mismatch (0-based vs 1-based)
    """

    def try_load(path):
        if not os.path.exists(path): return None
        try:
            if grayscale:
                return np.array(Image.open(path).convert('L'))
            else:
                return np.array(Image.open(path).convert('RGB'))
        except:
            return None

    # 1. Try Original Path
    img = try_load(img_path)
    if img is not None: return img

    # 2. Fallback Logic
    # print(f"⚠️ [IO] Broken link: {os.path.basename(img_path)}")
    dir_name = os.path.dirname(img_path)
    file_name = os.path.basename(img_path)
    folder_type = os.path.basename(dir_name)  # 'frames' or 'masks'
    job_root = os.path.dirname(os.path.dirname(dir_name))

    # Define potential fallback directories
    fallback_dirs = [os.path.join(job_root, folder_type)]
    if folder_type == 'masks':
        fallback_dirs.append(os.path.join(job_root, 'masks'))

    # Regex to parse "frame_000123.png" -> number 123
    match = re.search(r'(\d+)', file_name)
    if not match:
        raise ValueError(f"CRITICAL: Cannot parse filename {file_name}")

    idx = int(match.group(1))
    prefix = file_name[:match.start(1)]
    suffix = file_name[match.end(1):]

    # Try indices: [original, +1, -1]
    # Because ffmpeg starts at 1, but python script might ask for 0
    indices_to_try = [idx, idx + 1, idx - 1]

    for d in fallback_dirs:
        for i in indices_to_try:
            # Reconstruct filename: frame_ + 000123 + .png
            # Assuming 6 digits padding is standard here based on logs
            candidate_name = f"{prefix}{i:06d}{suffix}"
            candidate_path = os.path.join(d, candidate_name)

            img = try_load(candidate_path)
            if img is not None:
                # print(f"✅ [IO] Recovered {file_name} -> {candidate_name}")
                return img

    raise ValueError(
        f"CRITICAL: Could not recover {file_name} (Checked dirs: {fallback_dirs}, indices: {indices_to_try})")


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
        return torch.from_numpy(flow_full).permute(2, 0, 1).unsqueeze(0)  # Keep on CPU for now

    for i in range(len(frames) - 1):
        flows_forward.append(calc_flow(gray_frames[i], gray_frames[i + 1]))
    for i in range(1, len(frames)):
        flows_backward.append(calc_flow(gray_frames[i], gray_frames[i - 1]))

    return torch.stack(flows_forward, dim=1), torch.stack(flows_backward, dim=1)


def run_pipeline(args):
    print("🚀 [Core] Starting ProPainter (Index Fix + FP16)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LOAD MODEL FP16
    print("📦 Loading Model...")
    model = InpaintGenerator(model_path=args.model_path).to(device).half().eval()

    f_files = sorted(
        [os.path.join(args.video, f) for f in os.listdir(args.video) if f.endswith(('.jpg', '.png', '.jpeg'))])
    m_files = sorted(
        [os.path.join(args.mask, f) for f in os.listdir(args.mask) if f.endswith(('.jpg', '.png', '.jpeg'))])

    if not f_files: raise ValueError("No frames found")
    print(f"🔄 Processing {len(f_files)} frames...")

    # 1. LOAD REFERENCE DIMS
    ref_img = smart_imread(f_files[0], grayscale=False)
    target_h, target_w = ref_img.shape[:2]

    def load_prep(path, gray=False):
        img = smart_imread(path, gray)
        if img.shape[:2] != (target_h, target_w):
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST if gray else cv2.INTER_LINEAR)

        img = pad_img_to_modulo(img, 16)
        if gray:
            img = (img > 127).astype(np.uint8) * 255
            return (torch.from_numpy(img).float() / 255.0 > 0.5).float().unsqueeze(0)
        else:
            return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    video_list, mask_list = [], []
    raw_frames_flow = []

    for f_p, m_p in zip(f_files, m_files):
        t_img = load_prep(f_p, False)
        t_msk = load_prep(m_p, True)
        video_list.append(t_img)
        mask_list.append(t_msk)
        raw_frames_flow.append((t_img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8))

    masked_frames_gpu = torch.stack(video_list).unsqueeze(0).to(device).half()
    masks_gpu = torch.stack(mask_list).unsqueeze(0).to(device).half()

    # FLOW
    flows_fwd_cpu, flows_bwd_cpu = compute_flow_opencv(raw_frames_flow)
    flows_fwd_gpu = flows_fwd_cpu.to(device).half()
    flows_bwd_gpu = flows_bwd_cpu.to(device).half()

    torch.cuda.empty_cache()

    # INFERENCE
    print("⚡ ProPainter Inference...")
    # Safe Context
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True):
        with torch.no_grad():
            pred = model(masked_frames_gpu, (flows_fwd_gpu, flows_bwd_gpu), masks_gpu)[0]

    # SAVE
    print("💾 Saving...")
    os.makedirs(args.output, exist_ok=True)

    for i, f_path in enumerate(f_files):
        p = pred[i].permute(1, 2, 0).float().cpu().numpy().clip(0, 1)
        p = p[:target_h, :target_w, :]

        # Load originals for comp using smart read
        orig = smart_imread(f_path).astype(float) / 255.0
        m = (smart_imread(m_files[i], True).astype(float) / 255.0 > 0.5)[:, :, None]

        if orig.shape[:2] != (target_h, target_w): orig = cv2.resize(orig, (target_w, target_h))
        if m.shape[:2] != (target_h, target_w): m = cv2.resize(m, (target_w, target_h))[:, :, None]

        final = p * m + orig * (1 - m)
        out_path = os.path.join(args.output, os.path.basename(f_path))
        Image.fromarray((final * 255).astype(np.uint8)).save(out_path)

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

    try:
        run_pipeline(args)
    except Exception as e:
        print(f"\n❌ FATAL: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)