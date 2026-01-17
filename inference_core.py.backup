"""
Optimized version of inference_core.py using PyTorch AMP (Automatic Mixed Precision)
and other memory optimizations.
"""

import os
import cv2
import torch
import argparse
import numpy as np
import warnings
import sys
import contextlib

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
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

def process_video_in_chunks(video_tensor, mask_tensor, model, flow_models, 
                           chunk_size=10, overlap=2, args=None):
    """
    Process long videos in chunks to save memory.
    
    Args:
        video_tensor: [1, T, C, H, W]
        mask_tensor: [1, T, 1, H, W]
        model: ProPainter model
        flow_models: Tuple of (raft_model, flow_complete_model)
        chunk_size: Number of frames per chunk
        overlap: Overlap between chunks
        args: Command line arguments
        
    Returns:
        Completed frames list
    """
    raft_model, flow_complete_model = flow_models
    device = video_tensor.device
    T = video_tensor.shape[1]
    
    # Determine optimal chunk size based on available memory
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        used_memory = torch.cuda.memory_allocated()
        available_memory = total_memory - used_memory
        
        # Estimate memory per frame (rough estimate)
        frame_memory = video_tensor[0, 0].numel() * video_tensor.element_size()
        max_frames_in_memory = int(available_memory * 0.7 / frame_memory)
        chunk_size = min(chunk_size, max_frames_in_memory)
        print(f"📊 Memory-aware chunking: {chunk_size} frames per chunk")
    
    results = []
    
    for start in range(0, T, chunk_size - overlap):
        end = min(start + chunk_size, T)
        
        # Calculate indices with overlap
        chunk_start = max(0, start - overlap)
        chunk_end = min(T, end + overlap)
        
        # Extract chunk
        video_chunk = video_tensor[:, chunk_start:chunk_end]
        mask_chunk = mask_tensor[:, chunk_start:chunk_end]
        
        print(f"🔧 Processing chunk {start//chunk_size + 1}/{(T + chunk_size - 1)//chunk_size}: "
              f"frames {chunk_start}:{chunk_end}")
        
        # Process chunk
        chunk_result = process_single_chunk(
            video_chunk, mask_chunk, model, raft_model, flow_complete_model, args
        )
        
        # Trim overlap
        result_start = start - chunk_start
        result_end = result_start + (end - start)
        trimmed_result = chunk_result[:, result_start:result_end]
        
        results.append(trimmed_result)
    
    return torch.cat(results, dim=1)

def process_single_chunk(video_tensor, mask_tensor, model, raft_model, 
                        flow_complete_model, args):
    """
    Process a single chunk of video.
    """
    device = video_tensor.device
    b, t, c, h, w = video_tensor.shape
    
    # Enable AMP for this chunk
    use_amp = torch.cuda.is_available()
    dtype = torch.float16 if use_amp else torch.float32
    
    with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', 
                       dtype=dtype, enabled=use_amp):
        
        # 1. Compute flows (keep RAFT in FP32 for stability)
        with torch.no_grad():
            gt_flows_bi = raft_model(video_tensor.float(), iters=args.raft_iter)
        
        # 2. Complete flows (FP32 for stability)
        with torch.no_grad():
            pred_flows_bi, _ = flow_complete_model.forward_bidirect_flow(
                gt_flows_bi, mask_tensor.float()
            )
            pred_flows_bi = flow_complete_model.combine_flow(
                gt_flows_bi, pred_flows_bi, mask_tensor.float()
            )
        
        # 3. Temporal propagation
        with torch.no_grad():
            prop_imgs, updated_local_masks = model.img_propagation(
                video_tensor * (1 - mask_tensor),
                pred_flows_bi,
                mask_tensor,
                'nearest'
            )
        
        updated_masks = updated_local_masks.view(b, t, 1, h, w)
        updated_frames = video_tensor * (1 - mask_tensor) + prop_imgs.view(b, t, c, h, w) * mask_tensor
        
        # 4. Final inference
        neighbor_length = min(t, args.neighbor_length)
        neighbor_stride = neighbor_length // 2
        
        comp_frames_tensor = torch.zeros_like(video_tensor)
        frame_weights = torch.zeros(b, t, 1, 1, 1, device=device)
        
        for f in range(0, t, neighbor_stride):
            neighbor_ids = [
                i for i in range(max(0, f - neighbor_stride),
                                 min(t, f + neighbor_stride + 1))
            ]
            
            # Use all frames as reference (simplified)
            ref_ids = []
            for i in range(0, t, args.ref_stride):
                if i not in neighbor_ids:
                    ref_ids.append(i)
            
            selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
            selected_masks = mask_tensor[:, neighbor_ids + ref_ids, :, :, :]
            selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
            
            # Handle flow selection
            if len(neighbor_ids) > 1:
                selected_pred_flows_bi = (
                    pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :],
                    pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :]
                )
            else:
                # Empty flow tensors
                selected_pred_flows_bi = (
                    pred_flows_bi[0][:, :0, :, :, :],
                    pred_flows_bi[1][:, :0, :, :, :]
                )
            
            with torch.no_grad():
                l_t = len(neighbor_ids)
                pred_img = model(selected_imgs, selected_pred_flows_bi, 
                               selected_masks, selected_update_masks, l_t)
                pred_img = pred_img.view(-1, c, h, w)
                
                # Blend with existing results
                for i, idx in enumerate(neighbor_ids):
                    comp_frames_tensor[:, idx] += pred_img[i:i+1]
                    frame_weights[:, idx] += 1
        
        # Average overlapping predictions
        comp_frames_tensor = comp_frames_tensor / frame_weights.clamp(min=1)
        
        return comp_frames_tensor

def main(args):
    print(f"🚀 [Core] Starting ProPainter Inference (Optimized AMP)...")
    print(f"   Video: {args.video}")
    print(f"   Mask:  {args.mask}")
    print(f"   Out:   {args.output}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load RAFT (Flow Estimation) - Keep in FP32 for stability
    print(f"📦 Loading RAFT from {args.raft_model_path}...")
    if not os.path.exists(args.raft_model_path):
        raise FileNotFoundError(f"RAFT weights not found at {args.raft_model_path}")
    fix_raft = RAFT_bi(args.raft_model_path, device)
    fix_raft.eval()
    
    # 2. Load Flow Completion Model - Keep in FP32
    print(f"📦 Loading Flow Completion from {args.fc_model_path}...")
    if not os.path.exists(args.fc_model_path):
        raise FileNotFoundError(f"Flow completion weights not found at {args.fc_model_path}")
    fix_flow_complete = RecurrentFlowCompleteNet(args.fc_model_path).to(device)
    fix_flow_complete.eval()
    
    # 3. Load ProPainter Model
    print(f"📦 Loading ProPainter from {args.model_path}...")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"ProPainter weights not found at {args.model_path}")
    model = InpaintGenerator(model_path=args.model_path).to(device)
    model.eval()
    
    # Enable torch.compile if available (PyTorch 2.x)
    if torch.cuda.is_available() and hasattr(torch, 'compile'):
        try:
            print("⚡ Enabling torch.compile for performance boost...")
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            print("✅ torch.compile enabled")
        except Exception as e:
            print(f"⚠️ torch.compile failed: {e}")
            print("⚠️ Falling back to eager mode")
    
    # 4. Prepare Data
    frames = sorted([os.path.join(args.video, f) for f in os.listdir(args.video) 
                     if f.endswith(('.jpg', '.png', '.jpeg'))])
    masks = sorted([os.path.join(args.mask, f) for f in os.listdir(args.mask) 
                    if f.endswith(('.jpg', '.png', '.jpeg'))])

    if len(frames) == 0:
        raise ValueError(f"No frames found in {args.video}")
    if len(frames) != len(masks):
        raise ValueError(f"Mismatch: {len(frames)} frames vs {len(masks)} masks")

    # Read reference for size
    ref_img = imread(frames[0])
    orig_h, orig_w = ref_img.shape[:2]
    print(f"📏 Resolution: {orig_w}x{orig_h}")
    print(f"📊 Total frames: {len(frames)}")

    os.makedirs(args.output, exist_ok=True)

    # 5. Load and preprocess all frames
    print("🔄 Pre-processing frames...")
    video_data = []
    mask_data = []

    for f_path, m_path in zip(frames, masks):
        img = imread(f_path)
        msk = cv2.imread(m_path, 0)
        
        # Force strict binary mask (0 or 255) to avoid gray artifacts
        msk = (msk > 127).astype(np.uint8) * 255

        # Resize/Pad logic (CRITICAL for CUDA stability)
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
    
    print(f"📦 Tensor shape: {video_tensor.shape}")
    print(f"💾 Memory usage: {video_tensor.numel() * video_tensor.element_size() / 1024**3:.2f} GB")
    
    # 6. Process video (with chunking for long videos)
    T = video_tensor.shape[1]
    if T > 50:  # Use chunking for long videos
        print(f"🔀 Using chunked processing for {T} frames...")
        comp_frames_tensor = process_video_in_chunks(
            video_tensor, mask_tensor, model, 
            (fix_raft, fix_flow_complete),
            chunk_size=args.chunk_size,
            overlap=2,
            args=args
        )
    else:
        print(f"🔀 Processing all {T} frames at once...")
        comp_frames_tensor = process_single_chunk(
            video_tensor, mask_tensor, model, fix_raft, fix_flow_complete, args
        )
    
    # 7. Save results
    print("💾 Saving results...")
    comp_frames = comp_frames_tensor[0].cpu().permute(0, 2, 3, 1).numpy()
    comp_frames = np.clip((comp_frames + 1) / 2, 0, 1) * 255  # Convert from [-1,1] to [0,255]
    
    for i, (frame_path, mask_path) in enumerate(zip(frames, masks)):
        # Read original image and mask for background preservation
        orig_img = imread(frame_path).astype(np.float32) / 255.0
        orig_mask = cv2.imread(mask_path, 0).astype(np.float32) / 255.0
        orig_mask = (orig_mask > 0.5)[:, :, None]
        
        # Get completed frame
        comp_frame = comp_frames[i].astype(np.float32) / 255.0
        
        # Blend with original background
        final_img = comp_frame * orig_mask + orig_img * (1 - orig_mask)
        
        # Save
        save_path = os.path.join(args.output, os.path.basename(frame_path))
        final_bgr = cv2.cvtColor((np.clip(final_img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, final_bgr)
        
        if (i + 1) % 10 == 0:
            print(f"   Saved {i + 1}/{len(frames)} frames")
    
    print(f"✅ Done. Results saved to {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimized ProPainter Inference with AMP')
    parser.add_argument('--video', type=str, required=True, help='Path to input frames folder')
    parser.add_argument('--mask', type=str, required=True, help='Path to input masks folder')
    parser.add_argument('--output', type=str, required=True, help='Path to output folder')
    parser.add_argument('--model_path', type=str, default='weights/ProPainter.pth', 
                       help='Path to ProPainter .pth model')
    parser.add_argument('--raft_model_path', type=str, default='weights/raft-things.pth', 
                       help='Path to RAFT .pth model')
    parser.add_argument('--fc_model_path', type=str, default='weights/recurrent_flow_completion.pth', 
                       help='Path to flow completion .pth model')
    parser.add_argument('--raft_iter', type=int, default=20, help='RAFT iterations')
    parser.add_argument('--ref_stride', type=int, default=10, help='Reference frame stride')
    parser.add_argument('--neighbor_length', type=int, default=20, help='Neighbor window length')
    parser.add_argument('--chunk_size', type=int, default=10, 
                       help='Number of frames to process at once (for memory optimization)')
    args = parser.parse_args()
    main(args)
