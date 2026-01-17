import os
import sys
import cv2
import torch
import argparse
import numpy as np
import warnings
import torch.nn.functional as F
import types

# Add repo root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 1. IMPORT MODELS
from model.propainter import InpaintGenerator
from model.modules.flow_comp_raft import RAFT_bi

# Import the module where CorrBlock lives (adjust path if needed based on repo structure)
try:
    import model.modules.RAFT.core.corr as raft_corr_module
except ImportError:
    try:
        import RAFT.core.corr as raft_corr_module
    except ImportError:
        raft_corr_module = None
        print("⚠️ [Hot-Patch] Could not find RAFT.core.corr module to patch. Proceeding with caution.")


# 2. RUNTIME HOT-PATCH: Define Safe CorrBlock
class SafeCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # Safe MatMul (FP32 + Contiguous)
        batch, dim, ht, wd = fmap1.shape
        f1 = fmap1.view(batch, dim, ht * wd).transpose(1, 2)
        f2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(f1.float(), f2.float())
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        corr = corr / torch.sqrt(torch.tensor(dim).float())

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            # Safe Meshgrid
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1).to(coords.device)
            delta = delta.flip(-1)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            # Safe Grid Sample
            H, W = corr.shape[-2:]
            x, y = coords_lvl.unbind(-1)
            x = 2 * (x / (W - 1)) - 1
            y = 2 * (y / (H - 1)) - 1
            grid = torch.stack([x, y], dim=-1)

            sample = F.grid_sample(corr, grid, align_corners=True, padding_mode='border')
            sample = sample.view(batch, h1, w1, -1)
            out_pyramid.append(sample)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()


# Apply Hot-Patch
if raft_corr_module:
    print(f"🛡️ [Hot-Patch] Injecting SafeCorrBlock into {raft_corr_module.__name__}...")
    raft_corr_module.CorrBlock = SafeCorrBlock
    raft_corr_module.alt_cuda_corr = None  # Disable C++ extension

# 3. MAIN SCRIPT
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


def main(args):
    print("🚀 [Core] Starting ProPainter (Runtime Hot-Patch + CPU RAFT)...")

    # Devices
    cpu_dev = torch.device("cpu")
    gpu_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Models
    base = os.path.dirname(os.path.abspath(__file__))
    raft_p = args.raft_model_path or os.path.join(base, 'weights', 'raft-things.pth')
    pp_p = args.model_path

    # Init RAFT on CPU
    print("📦 Loading RAFT on CPU...")
    try:
        fix_raft = RAFT_bi(model_path=raft_p, device=cpu_dev)
    except Exception as e:
        print(f"❌ Error loading RAFT: {e}")
        print("   (Proceeding might fail if RAFT is required)")
        raise e

    # Init ProPainter on GPU
    print("📦 Loading ProPainter on GPU...")
    model = InpaintGenerator(model_path=pp_p).to(gpu_dev).eval()

    # Data
    frames = sorted(
        [os.path.join(args.video, f) for f in os.listdir(args.video) if f.endswith(('.jpg', '.png', '.jpeg'))])
    masks = sorted([os.path.join(args.mask, f) for f in os.listdir(args.mask) if f.endswith(('.jpg', '.png', '.jpeg'))])

    video_data = []
    mask_data = []
    for f, m in zip(frames, masks):
        img = pad_img_to_modulo(imread(f), 16)
        msk = pad_img_to_modulo((cv2.imread(m, 0) > 127).astype(np.uint8) * 255, 16)
        video_data.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
        mask_data.append((torch.from_numpy(msk).float() / 255.0 > 0.5).float().unsqueeze(0))

    masked_frames_gpu = torch.stack(video_data).unsqueeze(0).to(gpu_dev)
    masks_gpu = torch.stack(mask_data).unsqueeze(0).to(gpu_dev)

    # --- RAFT EXECUTION (CPU) ---
    print("🌊 Running RAFT on CPU (0.5x scale)...")
    scale = 0.5
    b, t, c, h, w = masked_frames_gpu.shape
    h_s, w_s = int(h * scale), int(w * scale)

    # Downscale on GPU -> Move to CPU
    vid_s_gpu = F.interpolate(masked_frames_gpu.view(-1, c, h, w), size=(h_s, w_s), mode='bilinear',
                              align_corners=False)
    vid_s_cpu = vid_s_gpu.view(b, t, c, h_s, w_s).cpu()

    with torch.no_grad():
        # Ensure inputs are contiguous on CPU
        vid_s_cpu = vid_s_cpu.contiguous()
        # Execute (using Hot-Patched Block)
        flows_s_cpu = fix_raft(vid_s_cpu, None)

    print("🌊 RAFT Done. Upscaling on GPU...")
    flows_l = []
    for f in flows_s_cpu:
        f_gpu = f.to(gpu_dev)
        up = F.interpolate(f_gpu.view(-1, 2, h_s, w_s), size=(h, w), mode='bilinear', align_corners=False)
        flows_l.append(up.view(b, t - 1, 2, h, w) * (1.0 / scale))
    gt_flows = tuple(flows_l)

    del vid_s_cpu, flows_s_cpu
    torch.cuda.empty_cache()

    # --- INFERENCE ---
    print("⚡ ProPainter Inference...")
    from model.recurrent_flow_completion import RecurrentFlowCompleteNet
    fc_p = args.fc_model_path or os.path.join(base, 'weights', 'recurrent_flow_completion.pth')
    fix_fc = RecurrentFlowCompleteNet(fc_p).to(gpu_dev).eval()

    with torch.no_grad():
        updated_flows = fix_fc(gt_flows[0], gt_flows[1], masks_gpu)
        pred = model(masked_frames_gpu, updated_flows, masks_gpu)[0]

    # Save
    print("💾 Saving...")
    os.makedirs(args.output, exist_ok=True)
    orig_h, orig_w = imread(frames[0]).shape[:2]
    for i, frame in enumerate(frames):
        p = pred[i].permute(1, 2, 0).float().cpu().numpy().clip(0, 1)
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