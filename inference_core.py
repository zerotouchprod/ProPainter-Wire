import os
import sys
import time


# --- SELF-HEALING BLOCK: PATCH CORR.PY BEFORE IMPORTS ---
def patch_raft_correlation():
    """
    Finds and overwrites corr.py with a safe, pure-PyTorch implementation
    to prevent CUDA 12/RTX 40 series crashes.
    """
    print("🛡️ [Self-Heal] Searching for 'corr.py' to patch...")

    # Locate the file
    target_file = None
    search_root = os.path.dirname(os.path.abspath(__file__))
    for root, dirs, files in os.walk(search_root):
        if "corr.py" in files and "RAFT" in root:
            target_file = os.path.join(root, "corr.py")
            break

    if not target_file:
        print("⚠️ [Self-Heal] Could not find 'corr.py'. Skipping patch.")
        return

    print(f"🛡️ [Self-Heal] Patching {target_file}...")

    # The Safe Code (Pure PyTorch, No custom CUDA)
    safe_code = """
import torch
import torch.nn.functional as F

# FORCE DISABLE CUSTOM CUDA
alt_cuda_corr = None

class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # Force FP32 + Contiguous
        corr = CorrBlock.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dy = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1).to(coords.device)
            delta = delta.flip(-1)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            # Manual Bilinear Sample to avoid import deps
            # Normalize to [-1, 1]
            H, W = corr.shape[-2:]
            xgrid, ygrid = coords_lvl.split([1,1], dim=-1)
            xgrid = 2*xgrid/(W-1) - 1
            ygrid = 2*ygrid/(H-1) - 1
            grid = torch.cat([xgrid, ygrid], dim=-1)

            sample = F.grid_sample(corr, grid, align_corners=True)
            sample = sample.view(batch, h1, w1, -1)
            out_pyramid.append(sample)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd)

        # SAFE MATMUL: FP32 + CONTIGUOUS
        f1 = fmap1.transpose(1,2).float().contiguous()
        f2 = fmap2.float().contiguous()
        corr = torch.matmul(f1, f2)

        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())

class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.block = CorrBlock(fmap1, fmap2, num_levels, radius)
    def __call__(self, coords):
        return self.block(coords)
"""
    # Overwrite file
    with open(target_file, "w") as f:
        f.write(safe_code)

    # Delete __pycache__ to force reload
    cache_dir = os.path.join(os.path.dirname(target_file), "__pycache__")
    if os.path.exists(cache_dir):
        import shutil
        try:
            shutil.rmtree(cache_dir)
            print("🛡️ [Self-Heal] Cleared __pycache__.")
        except:
            pass

    print("✅ [Self-Heal] Patch applied successfully.")


# EXECUTE PATCH IMMEDIATELY
patch_raft_correlation()
# --------------------------------------------------------

import cv2
import torch
import argparse
import numpy as np
import warnings
import torch.nn.functional as F

# Add repo root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.propainter import InpaintGenerator
from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet

warnings.filterwarnings("ignore")

# Disable TF32 to prevent precision-related crashes on Ampere+
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


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
    print(f"🚀 [Core] Starting ProPainter (Self-Healed + Safe Mode)...")
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

    # RAFT
    print("🌊 Running RAFT (Smart Downscale)...")
    scale = 0.5
    b, t, c, h, w = masked_frames.shape
    h_s, w_s = int(h * scale), int(w * scale)

    vid_s = F.interpolate(masked_frames.view(-1, c, h, w), size=(h_s, w_s), mode='bilinear', align_corners=False)
    vid_s = vid_s.view(b, t, c, h_s, w_s)

    with torch.no_grad():
        vid_s = vid_s.contiguous()
        # This call uses the patched CorrBlock now
        flows_s = fix_raft(vid_s, None)

    # Upscale
    flows_l = []
    for f in flows_s:
        up = F.interpolate(f.view(-1, 2, h_s, w_s), size=(h, w), mode='bilinear', align_corners=False)
        flows_l.append(up.view(b, t - 1, 2, h, w) * (1.0 / scale))
    gt_flows = tuple(flows_l)

    del vid_s, flows_s
    torch.cuda.empty_cache()

    # Inference
    print("⚡ Inference...")
    with torch.no_grad():
        updated_flows = fix_fc(gt_flows[0], gt_flows[1], masks_tensor)
        in_v = masked_frames.half() if use_half else masked_frames
        in_f = (updated_flows[0].half(), updated_flows[1].half()) if use_half else updated_flows
        in_m = masks_tensor.half() if use_half else masks_tensor
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