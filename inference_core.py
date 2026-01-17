import os
import sys
import cv2
import torch
import argparse
import numpy as np
import warnings
import re
from PIL import Image

# PARANOID STABILITY
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.propainter import InpaintGenerator

# DEFAULT ROI FALLBACK (From your pipeline command)
# x, y, w, h (relative 0.0-1.0)
FALLBACK_ROI = (0.05, 0.5, 0.9, 0.3)


def smart_imread(img_path, grayscale=False):
    """ Reads image with symlink/index recovery. Returns None if failed. """

    def try_load(path):
        if not os.path.exists(path): return None
        try:
            if grayscale:
                return np.array(Image.open(path).convert('L'))
            else:
                return np.array(Image.open(path).convert('RGB'))
        except:
            return None

    # 1. Direct
    img = try_load(img_path)
    if img is not None: return img

    # 2. Fallback Search
    dir_name = os.path.dirname(img_path)
    file_name = os.path.basename(img_path)
    folder_type = os.path.basename(dir_name)
    job_root = os.path.dirname(os.path.dirname(dir_name))

    fallback_dirs = [os.path.join(job_root, folder_type)]
    if folder_type == 'masks': fallback_dirs.append(os.path.join(job_root, 'masks'))

    # Parse Index
    try:
        match = re.search(r'(\d+)', file_name)
        if match:
            idx = int(match.group(1))
            prefix = file_name[:match.start(1)]
            suffix = file_name[match.end(1):]
            indices_to_try = [idx, idx + 1, idx - 1]

            for d in fallback_dirs:
                for i in indices_to_try:
                    for padding in [6, 5, 4]:  # Try different zero-padding
                        name = f"{prefix}{i:0{padding}d}{suffix}"
                        img = try_load(os.path.join(d, name))
                        if img is not None: return img
    except:
        pass

    return None  # Failed to find file


def generate_fallback_mask(h, w):
    """ Generates a synthetic mask based on default ROI """
    mask = np.zeros((h, w), dtype=np.uint8)

    x_rel, y_rel, w_rel, h_rel = FALLBACK_ROI
    x = int(x_rel * w)
    y = int(y_rel * h)
    mw = int(w_rel * w)
    mh = int(h_rel * h)

    # Draw white rectangle (255)
    cv2.rectangle(mask, (x, y), (x + mw, y + mh), 255, -1)
    return mask


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
        return torch.from_numpy(flow_full).permute(2, 0, 1).unsqueeze(0)

    for i in range(len(frames) - 1):
        flows_forward.append(calc_flow(gray_frames[i], gray_frames[i + 1]))
    for i in range(1, len(frames)):
        flows_backward.append(calc_flow(gray_frames[i], gray_frames[i - 1]))

    return torch.stack(flows_forward, dim=1), torch.stack(flows_backward, dim=1)


def run_pipeline(args):
    print("🚀 [Core] Starting ProPainter (Auto-Mask Gen Mode)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("📦 Loading Model (FP16)...")
    model = InpaintGenerator(model_path=args.model_path).to(device).half().eval()

    f_files = sorted(
        [os.path.join(args.video, f) for f in os.listdir(args.video) if f.endswith(('.jpg', '.png', '.jpeg'))])
    m_files = sorted(
        [os.path.join(args.mask, f) for f in os.listdir(args.mask) if f.endswith(('.jpg', '.png', '.jpeg'))])

    if not f_files: raise ValueError("No frames found")
    print(f"🔄 Processing {len(f_files)} frames...")

    # 1. SETUP DIMS
    # Must find at least one valid frame to set dimensions
    ref_img = smart_imread(f_files[0], grayscale=False)
    if ref_img is None:
        # Try finding ANY valid frame
        for f in f_files:
            ref_img = smart_imread(f, grayscale=False)
            if ref_img is not None: break
        if ref_img is None:
            raise ValueError("CRITICAL: All frame files are missing/broken.")

    orig_h, orig_w = ref_img.shape[:2]

    # OOM Protection
    MAX_WIDTH = 1280
    scale = 1.0
    if orig_w > MAX_WIDTH:
        scale = MAX_WIDTH / orig_w

    target_w = int(orig_w * scale)
    target_h = int(orig_h * scale)
    # Ensure divisible by 2
    target_w = (target_w // 2) * 2
    target_h = (target_h // 2) * 2

    print(f"📉 Resizing {orig_w}x{orig_h} -> {target_w}x{target_h}")

    # 2. LOAD DATA
    def load_prep(path, gray=False, is_mask=False):
        img = smart_imread(path, gray)

        # SYNTHETIC MASK FALLBACK
        if img is None and is_mask:
            # print(f"⚠️ Generating synthetic mask for {os.path.basename(path)}")
            img = generate_fallback_mask(target_h, target_w)  # Already resized
            # No resize needed below since we generated at target size
        elif img is None and not is_mask:
            raise ValueError(f"CRITICAL: Missing frame {path}")
        else:
            # Resize real image
            if img.shape[:2] != (target_h, target_w):
                img = cv2.resize(img, (target_w, target_h),
                                 interpolation=cv2.INTER_NEAREST if gray else cv2.INTER_LINEAR)

        img = pad_img_to_modulo(img, 16)

        if gray:
            img = (img > 127).astype(np.uint8) * 255
            return (torch.from_numpy(img).float() / 255.0 > 0.5).float().unsqueeze(0)
        else:
            return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    video_list, mask_list = [], []
    raw_frames_flow = []

    for f_p, m_p in zip(f_files, m_files):
        t_img = load_prep(f_p, False, is_mask=False)
        t_msk = load_prep(m_p, True, is_mask=True)
        video_list.append(t_img)
        mask_list.append(t_msk)
        raw_frames_flow.append((t_img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8))

    masked_frames_gpu = torch.stack(video_list).unsqueeze(0).to(device).half()
    masks_gpu = torch.stack(mask_list).unsqueeze(0).to(device).half()

    # 3. FLOW
    flows_fwd_cpu, flows_bwd_cpu = compute_flow_opencv(raw_frames_flow)
    flows_fwd_gpu = flows_fwd_cpu.to(device).half()
    flows_bwd_gpu = flows_bwd_cpu.to(device).half()

    torch.cuda.empty_cache()

    # 4. INFERENCE
    print("⚡ Inference...")
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True):
        with torch.no_grad():
            pred = model(masked_frames_gpu, (flows_fwd_gpu, flows_bwd_gpu), masks_gpu)[0]

    # 5. SAVE
    print("💾 Saving...")
    os.makedirs(args.output, exist_ok=True)

    for i, f_path in enumerate(f_files):
        p = pred[i].permute(1, 2, 0).float().cpu().numpy().clip(0, 1)
        p = p[:target_h, :target_w, :]

        # Upscale result back to original
        if (target_h, target_w) != (orig_h, orig_w):
            p = cv2.resize(p, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Compositing
        orig = smart_imread(f_path).astype(float) / 255.0

        # Get mask (synthetic or real)
        m_img = smart_imread(m_files[i], True)
        if m_img is None:
            # Re-generate synthetic mask at ORIGINAL size
            m_img = generate_fallback_mask(orig_h, orig_w)

        m = (m_img.astype(float) / 255.0 > 0.5)[:, :, None]

        # Ensure dims match
        if orig.shape[:2] != (orig_h, orig_w): orig = cv2.resize(orig, (orig_w, orig_h))
        if m.shape[:2] != (orig_h, orig_w): m = cv2.resize(m, (orig_w, orig_h))[:, :, None]

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