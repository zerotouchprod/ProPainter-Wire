import os
import sys

# THE SAFE CODE (Pure PyTorch, CPU/GPU Safe)
SAFE_CORR_CONTENT = """
import torch
import torch.nn.functional as F

# DISABLE CUSTOM CUDA
alt_cuda_corr = None

class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # Standard MatMul Correlation (Safe)
        batch, dim, ht, wd = fmap1.shape
        f1 = fmap1.view(batch, dim, ht*wd).transpose(1, 2)
        f2 = fmap2.view(batch, dim, ht*wd)

        # Force Float32 for stability
        corr = torch.matmul(f1.float(), f2.float())
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        corr = corr / torch.sqrt(torch.tensor(dim).float())

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

            # Manual Grid Sample (No utils dependency)
            H, W = corr.shape[-2:]
            x, y = coords_lvl.unbind(-1)
            x = 2*(x/(W-1)) - 1
            y = 2*(y/(H-1)) - 1
            grid = torch.stack([x, y], dim=-1)

            sample = F.grid_sample(corr, grid, align_corners=True, padding_mode='border')
            sample = sample.view(batch, h1, w1, -1)
            out_pyramid.append(sample)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

# Redirect Alternate to Standard
class AlternateCorrBlock(CorrBlock):
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        super().__init__(fmap1, fmap2, num_levels, radius)
"""


def main():
    print("🛡️ [Omni-Patch] Scanning for corr.py...")
    root_dir = "/opt/ProPainter-Wire"
    patched_count = 0

    for root, dirs, files in os.walk(root_dir):
        if "corr.py" in files:
            target = os.path.join(root, "corr.py")
            print(f"   -> Patching: {target}")
            with open(target, "w") as f:
                f.write(SAFE_CORR_CONTENT)

            # Nuke __pycache__
            pycache = os.path.join(root, "__pycache__")
            if os.path.exists(pycache):
                import shutil
                shutil.rmtree(pycache)
                print(f"      (Cleared __pycache__)")

            patched_count += 1

    if patched_count == 0:
        print("❌ ERROR: No corr.py files found! Check directory structure.")
        sys.exit(1)

    print(f"✅ [Omni-Patch] Successfully patched {patched_count} files.")

    # --- VERIFICATION TEST ---
    print("\n🧪 [Test] Verifying RAFT Stability...")
    try:
        sys.path.append(root_dir)
        # Import RAFT modules dynamically
        from model.modules.flow_comp_raft import RAFT_bi

        # Init RAFT on CPU
        print("   -> Initializing RAFT on CPU...")
        raft = RAFT_bi(None, device='cpu')
        # (Pass None as model_path just to test init, or real path if available)

        print("   -> Creating dummy inputs...")
        # RAFT expects [B, T, C, H, W]
        dummy_video = torch.randn(1, 3, 3, 256, 256)

        print("   -> Running Forward Pass (CPU)...")
        # We expect a failure loading weights (since passed None),
        # BUT we check if it crashes with CUDA error before that.
        # Actually, let's just check if CorrBlock can be instantiated.

        fmap1 = torch.randn(1, 128, 32, 32)
        fmap2 = torch.randn(1, 128, 32, 32)

        # Try to use the patched class
        # We need to find where it is imported.
        # Usually model.modules.RAFT.core.corr
        from model.modules.RAFT.core.corr import CorrBlock
        block = CorrBlock(fmap1, fmap2)
        res = block(torch.randn(1, 32, 32, 2))

        print(f"   -> CorrBlock Output: {res.shape}")
        print("✅ [Test] PASSED. Safe Core is active.")

    except Exception as e:
        print(f"❌ [Test] FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()