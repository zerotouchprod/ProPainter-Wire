import os
import sys
import cv2
import torch
import argparse
import numpy as np
import warnings
import re
import inspect
from PIL import Image

# STABILITY SETTINGS
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.propainter import InpaintGenerator

MAX_WIDTH = 1280


def smart_imread(img_path, grayscale=False):
    def try_load(path):
        if not os.path.exists(path): return None
        try:
            if grayscale:
                return np.array(Image.open(path).convert('L'))
            else:
                return np.array(Image.open(path).convert('RGB'))
        except:
            return None

    img = try_load(img_path)
    if img is not None: return img

    dir_name = os.path.dirname(img_path)
    file_name = os.path.basename(img_path)
    folder_type = os.path.basename(dir_name)
    job_root = os.path.dirname(os.path.dirname(dir_name))

    fallback_dirs = [os.path.join(job_root, folder_type)]
    if folder_type == 'masks': fallback_dirs.append(os.path.join(job_root, 'masks'))

    try:
        match = re.search(r'(\d+)', file_name)
        if match:
            idx = int(match.group(1))
            prefix = file_name[:match.start(1)]
            suffix = file_name[match.end(1):]
            indices = [idx, idx + 1, idx - 1]
            for d in fallback_dirs:
                for i in indices:
                    for p in [6, 5, 4]:
                        name = f"{prefix}{i:0{p}d}{suffix}"
                        img = try_load(os.path.join(d, name))
                        if img is not None: return img
    except:
        pass
    return None


def generate_fallback_mask(h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    x, y = int(0.05 * w), int(0.5 * h)
    mw, mh = int(0.9 * w), int(0.3 * h)
    cv2.rectangle(mask, (x, y), (x + mw, y + mh), 255, -1)
    return mask


def pad_img_to_modulo(img, mod):
    h, w = img.shape[:2]
    h_pad = ((h + mod - 1) // mod) * mod - h
    w_pad = ((w + mod - 1) // mod) * mod - w
    return cv2.copyMakeBorder(img, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)


def compute_flow_opencv(frames, downscale_factor=0.5):
    h, w = frames[0].shape[:2]
    h_s, w_s = int(h * downscale_factor), int(w * downscale_factor)
    gray = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (w_s, h_s), interpolation=cv2.INTER_LINEAR)
        gray.append(g)

    fwd, bwd = [], []

    def calc(prev, curr):
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) * (1.0 / downscale_factor)
        return torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0)

    for i in range(len(frames) - 1): fwd.append(calc(gray[i], gray[i + 1]))
    for i in range(1, len(frames)): bwd.append(calc(gray[i], gray[i - 1]))
    return torch.stack(fwd, dim=1), torch.stack(bwd, dim=1)


def run_pipeline(args):
    print("🚀 [Core] Starting ProPainter...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InpaintGenerator(model_path=args.model_path).to(device).half().eval()

    f_files = sorted(
        [os.path.join(args.video, f) for f in os.listdir(args.video) if f.endswith(('.jpg', '.png', '.jpeg'))])
    m_files = sorted(
        [os.path.join(args.mask, f) for f in os.listdir(args.mask) if f.endswith(('.jpg', '.png', '.jpeg'))])
    if not f_files: raise ValueError("No frames")

    # Setup Dims
    ref_img = smart_imread(f_files[0])
    if ref_img is None:
        for f in f_files:
            ref_img = smart_imread(f)
            if ref_img is not None: break
    if ref_img is None: raise ValueError("CRITICAL: No valid frames found")

    orig_h, orig_w = ref_img.shape[:2]
    scale = 1.0
    if orig_w > MAX_WIDTH: scale = MAX_WIDTH / orig_w
    target_w = (int(orig_w * scale) // 2) * 2
    target_h = (int(orig_h * scale) // 2) * 2

    # Load Data
    video_list, mask_list = [], []
    raw_flow = []

    for f_p, m_p in zip(f_files, m_files):
        img = smart_imread(f_p)
        if img is None: raise ValueError(f"Missing frame {f_p}")
        if img.shape[:2] != (target_h, target_w):
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img_pad = pad_img_to_modulo(img, 16)
        t_img = torch.from_numpy(img_pad).permute(2, 0, 1).float() / 255.0
        video_list.append(t_img)
        raw_flow.append((t_img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8))

        msk = smart_imread(m_p, True)
        if msk is None: msk = generate_fallback_mask(target_h, target_w)
        if msk.shape[:2] != (target_h, target_w):
            msk = cv2.resize(msk, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        msk_pad = pad_img_to_modulo((msk > 127).astype(np.uint8) * 255, 16)
        t_msk = (torch.from_numpy(msk_pad).float() / 255.0 > 0.5).float().unsqueeze(0)
        mask_list.append(t_msk)

    frames_gpu = torch.stack(video_list).unsqueeze(0).to(device).half()
    masks_gpu = torch.stack(mask_list).unsqueeze(0).to(device).half()

    # Flow
    fwd_cpu, bwd_cpu = compute_flow_opencv(raw_flow)
    fwd_gpu = fwd_cpu.to(device).half()
    bwd_gpu = bwd_cpu.to(device).half()

    # Inference
    torch.cuda.empty_cache()
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True):
        with torch.no_grad():
            pred = model(frames_gpu, (fwd_gpu, bwd_gpu), masks_gpu, masks_gpu, 0)[0]

    # Save
    os.makedirs(args.output, exist_ok=True)
    for i, f_path in enumerate(f_files):
        # Result
        p = pred[i].permute(1, 2, 0).float().cpu().numpy().clip(0, 1)
        p = p[:target_h, :target_w, :]  # Crop padding
        if (target_h, target_w) != (orig_h, orig_w):
            p = cv2.resize(p, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Composition
        orig = smart_imread(f_path).astype(float) / 255.0
        m = smart_imread(m_files[i], True)
        if m is None: m = generate_fallback_mask(orig_h, orig_w)
        m = (m.astype(float) / 255.0 > 0.5)[:, :, None]

        if orig.shape[:2] != (orig_h, orig_w): orig = cv2.resize(orig, (orig_w, orig_h))
        if m.shape[:2] != (orig_h, orig_w): m = cv2.resize(m, (orig_w, orig_h))[:, :, None]

        final = p * m + orig * (1 - m)

        # CRITICAL FIX: Use the BASENAME from the INPUT ARGUMENT list
        # Even if the file was a broken symlink pointing elsewhere,
        # we must save the output with the name the orchestrator expects (e.g. frame_000000.png)
        target_filename = os.path.basename(f_files[i])
        out_path = os.path.join(args.output, target_filename)

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