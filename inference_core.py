import os
import sys
import cv2
import torch
import argparse
import numpy as np
import warnings
import re
from PIL import Image

# --- CONFIG ---
MAX_WIDTH = 1280  # Защита от OOM (12.5GB VRAM)
MIN_FRAMES = 5  # Минимальная длина чанка для ProPainter

# --- SETUP ENV ---
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.propainter import InpaintGenerator


def log(msg):
    print(f"[Core] {msg}", flush=True)


def smart_imread(img_path, grayscale=False):
    """ Robust Image Reader (Symlinks, Index Shift, Fallbacks) """

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

    # Fallback search
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
    log("   Generating synthetic mask (Fallback)")
    mask = np.zeros((h, w), dtype=np.uint8)
    # ROI: 0.05, 0.5, 0.9, 0.3
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
    log("Starting...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Files
    f_files = sorted(
        [os.path.join(args.video, f) for f in os.listdir(args.video) if f.endswith(('.jpg', '.png', '.jpeg'))])
    m_files = sorted(
        [os.path.join(args.mask, f) for f in os.listdir(args.mask) if f.endswith(('.jpg', '.png', '.jpeg'))])

    if not f_files: raise ValueError("No frames found")

    # Model
    model = Inpaint