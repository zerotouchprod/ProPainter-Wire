import os
import sys
import cv2
import torch
import argparse
import numpy as np
import warnings
from PIL import Image

# PARANOID STABILITY SETTINGS
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.propainter import InpaintGenerator


def smart_imread(img_path, grayscale=False):
    """
    Robust Image Reader.
    1. Tries to read from the given path.
    2. If fails (broken symlink), hunts for the file in parent 'frames' or 'masks' folders.
    """

    def try_load(path):
        if not os.path.exists(path): return None
        try:
            if grayscale:
                # Open as grayscale
                return np.array(Image.open(path).convert('L'))
            else:
                # Open as RGB
                return np.array(Image.open(path).convert('RGB'))
        except:
            return None

    # 1. Try Original Path
    img = try_load(img_path)
    if img is not None: return img

    # 2. Broken Symlink Fallback
    print(f"⚠️ [IO] Broken link detected: {img_path}")

    dir_name = os.path.dirname(img_path)  # .../chunk_000/frames
    file_name = os.path.basename(img_path)  # frame_000.png
    folder_type = os.path.basename(dir_name)  # 'frames' or 'masks'

    # Heuristic: Go up 2 levels (chunk_000 -> job_root)
    # Structure: job/chunk_000/frames -> job/frames
    job_root = os.path.dirname(os.path.dirname(dir_name))

    # Try direct parent folder
    fallback_1 = os.path.join(job_root, folder_type, file_name)
    img = try_load(fallback_1)
    if img is not None:
        print(f"✅ [IO] Recovered from: {fallback_1}")
        return img

    # Try one level deeper (sometimes masks are in job/masks/output?)
    # Or just 'masks' at root if folder_type was something else
    if folder_type == 'masks':
        fallback_2 = os.path.join(job_root, 'masks', file_name)
        img = try_load(fallback_2)
        if img is not None: return img

    # If all fails
    raise ValueError(f"CRITICAL: Could not recover file {file_name}. Checked: {img_path}, {fallback_1}")


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
        # Frames are already Numpy arrays [H,W,3] RGB
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (w_s, h_s), interpolation=cv2.INTER_LINEAR)
        gray_frames.append(g)

    flows_forward, flows_backward = [], []

    def calc_flow(prev, curr):
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_full = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        flow_full *= (1.0 / downscale_factor)
        return torch.from_numpy(flow_full).permute(2, 0, 1).float().contiguous().unsqueeze(0)

    for i in range(len(frames) - 1):
        flows_forward.append(calc_flow(gray_frames[i], gray_frames[i + 1]))
    for i in range(1, len(frames)):
        flows_backward.append(calc_flow(gray_frames[i], gray_frames[i - 1]))

    return torch.stack(flows_forward, dim=1), torch.stack(flows_backward, dim=1)


def run_pipeline(args):
    print("🚀 [Core] Starting ProPainter (Smart-IO + Masks Fix)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("📦 Loading Model (FP32)...")
    model = InpaintGenerator(model_path=args.model_path).to(device).float().eval()

    f_files = sorted(
        [os.path.join(args.video, f) for f in os.listdir(args.video) if f.endswith(('.jpg', '.png', '.jpeg'))])
    m_files = sorted(
        [os.path.join(args.mask, f) for f in os.listdir(args.mask) if f.endswith(('.jpg', '.png', '.jpeg'))])

    if not f_files: raise ValueError("No frames found")
    print(f"🔄 Processing {len(f_files)} frames...")

    # 1. LOAD FRAMES (Smart)
    raw_frames = [pad_img_to_modulo(smart_imread(f, grayscale=False), 16) for f in f_files]

    video_list, mask_list = [], []
    for i, (img, m_path) in enumerate(zip(raw_frames, m_files)):
        # Frame Tensor
        t_img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        video_list.append(t_img.contiguous())

        # 2. LOAD MASKS (Smart)
        m_img = smart_imread(m_path, grayscale=True)
        m_img = pad_img_to_modulo((m_img > 127).astype(np.uint8) * 255, 16)

        t_msk = (torch.from_numpy(m_img).float() / 255.0 > 0.5).float().unsqueeze(0)
        mask_list.append(t_msk.contiguous())

    masked_frames_gpu = torch.stack(video_list).unsqueeze(0).to(device).contiguous()
    masks_gpu = torch.stack(mask_list).unsqueeze(0).to(device).contiguous()

    # 3. FLOW
    flows_fwd_cpu, flows_bwd_cpu = compute_flow_opencv(raw_frames)
    flows_fwd_gpu = flows_fwd_cpu.to(device).contiguous()
    flows_bwd_gpu = flows_bwd_cpu.to(device).contiguous()

    # 4. INFERENCE (Safe Attention)
    print("⚡ ProPainter Inference...")
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        with torch.no_grad():
            pred = model(masked_frames_gpu, (flows_fwd_gpu, flows_bwd_gpu), masks_gpu)[0]

    # 5. SAVE
    print("💾 Saving...")
    os.makedirs(args.output, exist_ok=True)

    # Get original size from the first successfully loaded frame
    # (Since raw_frames are padded, we need to unpad carefully or just crop)
    # The safest way is to crop to the size of the 'real' image on disk.
    # But for speed, let's use the first frame's original dims (before padding)
    # Wait, smart_imread returns unpadded. pad_img_to_modulo adds padding.
    # We can get original size from smart_imread(f_files[0]) again?
    # Or just calc it from raw_frames[0] minus padding?
    # Simpler: Load one original frame to get dims.
    ref_img = smart_imread(f_files[0], grayscale=False)
    orig_h, orig_w = ref_img.shape[:2]

    for i, f_path in enumerate(f_files):
        p = pred[i].permute(1, 2, 0).cpu().numpy().clip(0, 1)
        p = p[:orig_h, :orig_w, :]

        # We need the original image for background compositing
        # Since we already have raw_frames (padded), we can unpad it
        orig_padded = raw_frames[i].astype(float) / 255.0
        orig = orig_padded[:orig_h, :orig_w, :]

        # Same for mask
        m_padded = (mask_list[i].cpu().numpy().squeeze() > 0.5).astype(float)[:, :, None]
        m = m_padded[:orig_h, :orig_w, :]

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
    # Ignored
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