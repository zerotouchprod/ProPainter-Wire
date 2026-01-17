import os
import cv2
import torch
import argparse
import numpy as np
import warnings
import sys

# Добавляем текущую директорию в путь, чтобы импорты из model/ работали
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.propainter import InpaintGenerator

# Suppress annoying warnings
warnings.filterwarnings("ignore")

def imread(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def pad_img_to_modulo(img, mod):
    """
    Pad image to be divisible by mod (usually 8 or 16).
    Fixes CUBLAS errors on modern GPUs.
    """
    h, w = img.shape[:2]
    h_pad = ((h + mod - 1) // mod) * mod - h
    w_pad = ((w + mod - 1) // mod) * mod - w
    # Use REFLECT padding to avoid sharp edges at borders
    return cv2.copyMakeBorder(img, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)

def main(args):
    print(f"🚀 [Core] Starting ProPainter Inference...")
    print(f"   Video: {args.video}")
    print(f"   Mask:  {args.mask}")
    print(f"   Out:   {args.output}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    print(f"📦 Loading model from {args.model_path}...")
    if not os.path.exists(args.model_path):
         raise FileNotFoundError(f"Model weights not found at {args.model_path}")

    model = InpaintGenerator(model_path=args.model_path).to(device)
    model.eval()

    # 2. Precision Optimization (FP16)
    use_half = False
    if torch.cuda.is_available():
        try:
            # Try to switch to half precision
            model = model.half()
            use_half = True
            print("✅ Precision: FP16 (Half) enabled [VRAM Optimized]")
        except Exception as e:
            print(f"⚠️ FP16 failed, falling back to FP32. Error: {e}")
            model = model.float()
    else:
        print("⚠️ Running on CPU (Slow!)")

    # 3. Prepare Data
    frames = sorted([os.path.join(args.video, f) for f in os.listdir(args.video) if f.endswith(('.jpg', '.png', '.jpeg'))])
    masks = sorted([os.path.join(args.mask, f) for f in os.listdir(args.mask) if f.endswith(('.jpg', '.png', '.jpeg'))])

    if len(frames) == 0:
        raise ValueError(f"No frames found in {args.video}")
    if len(frames) != len(masks):
        raise ValueError(f"Mismatch: {len(frames)} frames vs {len(masks)} masks")

    # Read reference for size
    ref_img = imread(frames[0])
    orig_h, orig_w = ref_img.shape[:2]
    print(f"📏 Resolution: {orig_w}x{orig_h}")

    os.makedirs(args.output, exist_ok=True)

    # 4. Processing Loop (Chunk-based loading to save RAM)
    # Note: For very long videos, we should process in sub-batches.
    # Assuming the input `frames` folder is already a manageable chunk (e.g. 100 frames).
    
    video_data = []
    mask_data = []

    print("🔄 Pre-processing frames...")
    for f_path, m_path in zip(frames, masks):
        img = imread(f_path)
        msk = cv2.imread(m_path, 0)
        
        # Force strict binary mask (0 or 255) to avoid gray artifacts
        msk = (msk > 127).astype(np.uint8) * 255

        # Resize/Pad logic (CRITICAL for CUDA stability)
        # ProPainter requires dimensions divisible by 8 or 16
        img_padded = pad_img_to_modulo(img, 16)
        msk_padded = pad_img_to_modulo(msk, 16)

        img_t = torch.from_numpy(img_padded).permute(2, 0, 1).float() / 255.0
        msk_t = torch.from_numpy(msk_padded).float() / 255.0
        
        # Mask needs specific shape: [1, H, W]
        msk_t = (msk_t > 0.5).float().unsqueeze(0)

        video_data.append(img_t)
        mask_data.append(msk_t)

    # Stack to Tensor: [1, T, C, H, W]
    video_tensor = torch.stack(video_data).unsqueeze(0).to(device)
    mask_tensor = torch.stack(mask_data).unsqueeze(0).to(device)

    if use_half:
        video_tensor = video_tensor.half()
        mask_tensor = mask_tensor.half()

    # 5. Inference
    print("⚡ Running Inference (ProPainter)...")
    with torch.no_grad():
        try:
            # Main inference call
            pred_tensor = model(video_tensor, mask_tensor)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("❌ OOM Error! Suggestions: Reduce chunk size or use ROI cropping in the main service.")
                torch.cuda.empty_cache()
                raise e
            else:
                raise e

    # 6. Save & Post-process (Background Preservation)
    print("💾 Saving results...")
    pred_tensor = pred_tensor[0] # remove batch dim

    for i in range(len(frames)):
        # Convert back to numpy: [H_pad, W_pad, C]
        pred_frame = pred_tensor[i].permute(1, 2, 0)
        
        if use_half:
            pred_frame = pred_frame.float()
            
        pred_np = pred_frame.cpu().numpy()
        
        # Crop back to original size (remove padding)
        pred_np = pred_np[:orig_h, :orig_w, :]
        
        # Load original for background preservation
        # Why? ProPainter might slightly blur or color-shift unmasked areas.
        # We replace unmasked areas with the ORIGINAL pixel-perfect frame.
        orig_img = imread(frames[i]).astype(np.float32) / 255.0
        
        # Read mask again to determine what to keep
        orig_mask = cv2.imread(masks[i], 0).astype(np.float32) / 255.0
        orig_mask = (orig_mask > 0.5)[:, :, None] # [H, W, 1]

        # Composition: Inpainted * Mask + Original * (1 - Mask)
        final_img = pred_np * orig_mask + orig_img * (1 - orig_mask)
        
        # Save
        save_path = os.path.join(args.output, os.path.basename(frames[i]))
        final_bgr = cv2.cvtColor((np.clip(final_img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, final_bgr)

    print("✅ Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='Path to input frames folder')
    parser.add_argument('--mask', type=str, required=True, help='Path to input masks folder')
    parser.add_argument('--output', type=str, required=True, help='Path to output folder')
    parser.add_argument('--model_path', type=str, default='weights/ProPainter.pth', help='Path to .pth model')
    args = parser.parse_args()
    main(args)
