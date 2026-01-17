"""
Optimized ProPainter Inference Core with Smart Downscale, CPU Fallback, and Detailed Logging.
Priority: Stability over maximum performance.
Log Level: DEBUG by default for maximum visibility.
"""

import os
import cv2
import torch
import argparse
import numpy as np
import warnings
import sys
import time
import torch.nn.functional as F
from datetime import datetime
from typing import Tuple, List, Optional

# Add repo root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.propainter import InpaintGenerator
from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet

# Suppress warnings
warnings.filterwarnings("ignore")

class InferenceLogger:
    """Detailed logging with timestamps and memory monitoring"""
    
    def __init__(self, log_level: str = "DEBUG"):
        self.log_level = log_level
        self.start_time = time.time()
        self.memory_logs = []
        self.stage_times = {}
        
    def _should_log(self, level: str) -> bool:
        levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        return levels.get(level.upper(), 0) >= levels.get(self.log_level.upper(), 0)
    
    def log(self, level: str, message: str, emoji: str = ""):
        """Log message with timestamp and level"""
        if not self._should_log(level):
            return
            
        timestamp = datetime.now().strftime('%H:%M:%S')
        level_str = level.upper()
        prefix = f"[{timestamp}] {emoji} [{level_str}]" if emoji else f"[{timestamp}] [{level_str}]"
        print(f"{prefix} {message}")
        
        # Log memory if DEBUG level
        if level.upper() == "DEBUG" and torch.cuda.is_available():
            self.log_memory_snapshot()
    
    def log_memory_snapshot(self):
        """Log current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            self.memory_logs.append((time.time() - self.start_time, allocated, cached))
    
    def start_stage(self, stage_name: str):
        """Mark the beginning of a processing stage"""
        self.stage_times[stage_name] = time.time()
        self.log("INFO", f"Starting {stage_name}", "🔧")
    
    def end_stage(self, stage_name: str):
        """Mark the end of a processing stage and log duration"""
        if stage_name in self.stage_times:
            duration = time.time() - self.stage_times[stage_name]
            self.log("INFO", f"{stage_name} completed in {duration:.1f}s", "✅")
    
    def print_summary(self):
        """Print performance summary at the end"""
        total_time = time.time() - self.start_time
        self.log("INFO", f"Total inference time: {total_time:.1f}s", "📊")
        
        if self.memory_logs and torch.cuda.is_available():
            self.log("DEBUG", "Memory usage summary:", "💾")
            for t, allocated, cached in self.memory_logs[-5:]:  # Last 5 entries
                self.log("DEBUG", f"  t+{t:.1f}s: {allocated:.2f} GB allocated, {cached:.2f} GB cached")


def imread(img_path: str) -> np.ndarray:
    """Read image with proper error handling"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pad_img_to_modulo(img: np.ndarray, mod: int) -> np.ndarray:
    """Pad image to be divisible by mod (usually 8 or 16)"""
    h, w = img.shape[:2]
    h_pad = ((h + mod - 1) // mod) * mod - h
    w_pad = ((w + mod - 1) // mod) * mod - w
    return cv2.copyMakeBorder(img, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)


def calculate_optimal_scale_factor(h: int, w: int, logger: InferenceLogger) -> float:
    """
    Calculate optimal scale factor based on resolution.
    Returns scale factor between 0.125 and 1.0
    """
    total_pixels = h * w
    
    # Resolution-based scale factors
    if total_pixels > 3840 * 2160:  # > 8K
        scale = 0.125
        logger.log("DEBUG", f"Ultra-high resolution ({h}x{w}), using scale: {scale}x", "🌊")
    elif total_pixels > 1920 * 1080:  # > Full HD
        scale = 0.25
        logger.log("DEBUG", f"High resolution ({h}x{w}), using scale: {scale}x", "🌊")
    elif total_pixels > 1024 * 1024:  # > 1MP
        scale = 0.5
        logger.log("DEBUG", f"Medium resolution ({h}x{w}), using scale: {scale}x", "🌊")
    else:
        scale = 1.0
        logger.log("DEBUG", f"Low resolution ({h}x{w}), using original scale", "🌊")
    
    logger.log("INFO", f"Resolution: {h}x{w} ({total_pixels:,} pixels) -> Scale: {scale}x", "📏")
    return scale


def safe_raft_inference(
    video_tensor: torch.Tensor,
    raft_model: RAFT_bi,
    scale_factor: float,
    raft_iter: int,
    logger: InferenceLogger,
    enable_cpu_fallback: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Safe RAFT inference with downscale-upscale strategy and CPU fallback.
    
    Args:
        video_tensor: Input video tensor [B, T, C, H, W]
        raft_model: RAFT model
        scale_factor: Scale factor for downscaling (0.125-1.0)
        raft_iter: Number of RAFT iterations
        logger: InferenceLogger instance
        enable_cpu_fallback: Whether to enable CPU fallback on OOM
    
    Returns:
        Tuple of forward and backward flows
    """
    device = video_tensor.device
    b, t, c, h_orig, w_orig = video_tensor.shape
    
    logger.start_stage("RAFT Inference")
    
    # Try GPU inference first
    try:
        # Apply downscale if needed
        if scale_factor < 1.0:
            h_small = int(h_orig * scale_factor)
            w_small = int(w_orig * scale_factor)
            
            logger.log("DEBUG", f"Downscaling: {h_orig}x{w_orig} -> {h_small}x{w_small}", "📉")
            
            # Reshape for processing: [B, T, C, H, W] -> [B*T, C, H, W]
            video_reshaped = video_tensor.view(-1, c, h_orig, w_orig)
            
            # Downscale for RAFT computation
            video_small = F.interpolate(
                video_reshaped.float(),
                size=(h_small, w_small),
                mode='bilinear',
                align_corners=False
            )
            
            # Reshape back: [B*T, C, H_small, W_small] -> [B, T, C, H_small, W_small]
            video_small = video_small.view(b, t, c, h_small, w_small)
            
            # Run RAFT on downscaled video
            with torch.no_grad():
                flows_small = raft_model(video_small, iters=raft_iter)
            
            # Upscale flows back to original size
            flows_large = []
            for flow in flows_small:
                # flow shape: [B, T-1, 2, H_small, W_small]
                bf, tf, cf, hf, wf = flow.shape
                
                # Reshape for interpolation: [B*(T-1), 2, H_small, W_small]
                flow_flat = flow.view(-1, cf, hf, wf)
                
                # Upscale flow tensor
                upscaled = F.interpolate(
                    flow_flat,
                    size=(h_orig, w_orig),
                    mode='bilinear',
                    align_corners=False
                )
                
                # Scale flow values (optical flow scales with image size)
                upscaled = upscaled * (1.0 / scale_factor)
                
                # Reshape back: [B, T-1, 2, H_orig, W_orig]
                upscaled = upscaled.view(bf, tf, cf, h_orig, w_orig)
                flows_large.append(upscaled)
            
            gt_flows_bi = tuple(flows_large)
            
            # Clean up to free memory
            del video_small, flows_small, video_reshaped
            torch.cuda.empty_cache()
            
        else:
            # Original resolution is fine, use standard approach
            logger.log("DEBUG", "Using original resolution for RAFT", "🌊")
            with torch.no_grad():
                gt_flows_bi = raft_model(video_tensor.float(), iters=raft_iter)
        
        logger.end_stage("RAFT Inference")
        return gt_flows_bi
        
    except RuntimeError as e:
        error_msg = str(e).lower()
        logger.log("ERROR", f"RAFT inference failed with error: {e}", "❌")
        
        if "out of memory" in error_msg and enable_cpu_fallback:
            logger.log("WARNING", "GPU OOM detected, falling back to CPU inference", "⚠️")
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            
            # Move tensors and model to CPU
            video_cpu = video_tensor.cpu()
            raft_model_cpu = raft_model.cpu()
            
            try:
                # Try with even smaller scale on CPU (no recursion)
                smaller_scale = max(0.125, scale_factor * 0.5)
                logger.log("DEBUG", f"Trying smaller scale on CPU: {smaller_scale}x", "📉")
                
                # Simple CPU inference without recursion
                h_small = int(h_orig * smaller_scale)
                w_small = int(w_orig * smaller_scale)
                
                # Reshape and downscale
                video_reshaped = video_cpu.view(-1, c, h_orig, w_orig)
                video_small = F.interpolate(
                    video_reshaped.float(),
                    size=(h_small, w_small),
                    mode='bilinear',
                    align_corners=False
                )
                video_small = video_small.view(b, t, c, h_small, w_small)
                
                # Run RAFT on CPU
                with torch.no_grad():
                    flows_small = raft_model_cpu(video_small, iters=raft_iter)
                
                # Upscale flows
                flows_large = []
                for flow in flows_small:
                    bf, tf, cf, hf, wf = flow.shape
                    flow_flat = flow.view(-1, cf, hf, wf)
                    upscaled = F.interpolate(
                        flow_flat,
                        size=(h_orig, w_orig),
                        mode='bilinear',
                        align_corners=False
                    )
                    upscaled = upscaled * (1.0 / smaller_scale)
                    upscaled = upscaled.view(bf, tf, cf, h_orig, w_orig)
                    flows_large.append(upscaled)
                
                result = (flows_large[0].to(device), flows_large[1].to(device))
                
                logger.log("INFO", "CPU fallback successful", "✅")
                logger.end_stage("RAFT Inference")
                return result
                
            except Exception as cpu_error:
                logger.log("ERROR", f"CPU fallback also failed: {cpu_error}", "❌")
                # Re-raise the original GPU error for better debugging
                raise RuntimeError(f"GPU RAFT failed: {e}\nCPU fallback also failed: {cpu_error}")
        else:
            # Not an OOM error or fallback disabled
            raise RuntimeError(f"RAFT inference failed: {e}")


def process_video_in_chunks(
    video_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    model: InpaintGenerator,
    raft_model: RAFT_bi,
    flow_complete_model: RecurrentFlowCompleteNet,
    args,
    logger: InferenceLogger
) -> torch.Tensor:
    """
    Process long videos in chunks to save memory.
    
    Args:
        video_tensor: [1, T, C, H, W]
        mask_tensor: [1, T, 1, H, W]
        model: ProPainter model
        raft_model: RAFT model
        flow_complete_model: Flow completion model
        args: Command line arguments
        logger: InferenceLogger instance
    
    Returns:
        Completed frames tensor
    """
    device = video_tensor.device
    T = video_tensor.shape[1]
    chunk_size = args.chunk_size
    overlap = 2
    
    logger.log("INFO", f"Processing {T} frames in chunks of {chunk_size}", "🔀")
    
    # Determine optimal chunk size based on available memory
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        used_memory = torch.cuda.memory_allocated()
        available_memory = total_memory - used_memory
        
        # Estimate memory per frame (rough estimate)
        frame_memory = video_tensor[0, 0].numel() * video_tensor.element_size()
        max_frames_in_memory = int(available_memory * 0.7 / frame_memory)
        chunk_size = min(chunk_size, max_frames_in_memory)
        logger.log("DEBUG", f"Memory-aware chunking: {chunk_size} frames per chunk", "📊")
    
    results = []
    
    for start in range(0, T, chunk_size - overlap):
        end = min(start + chunk_size, T)
        
        # Calculate indices with overlap
        chunk_start = max(0, start - overlap)
        chunk_end = min(T, end + overlap)
        
        # Extract chunk
        video_chunk = video_tensor[:, chunk_start:chunk_end]
        mask_chunk = mask_tensor[:, chunk_start:chunk_end]
        
        chunk_num = start // (chunk_size - overlap) + 1
        total_chunks = (T + chunk_size - overlap - 1) // (chunk_size - overlap)
        logger.log("INFO", f"Processing chunk {chunk_num}/{total_chunks}: frames {chunk_start}:{chunk_end}", "🔧")
        
        # Process chunk
        chunk_result = process_single_chunk(
            video_chunk, mask_chunk, model, raft_model, flow_complete_model, args, logger
        )
        
        # Trim overlap
        result_start = start - chunk_start
        result_end = result_start + (end - start)
        trimmed_result = chunk_result[:, result_start:result_end]
        
        results.append(trimmed_result)
    
    return torch.cat(results, dim=1)


def process_single_chunk(
    video_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    model: InpaintGenerator,
    raft_model: RAFT_bi,
    flow_complete_model: RecurrentFlowCompleteNet,
    args,
    logger: InferenceLogger
) -> torch.Tensor:
    """
    Process a single chunk of video.
    """
    device = video_tensor.device
    b, t, c, h, w = video_tensor.shape
    
    logger.start_stage("Chunk Processing")
    
    # Enable AMP for this chunk
    use_amp = torch.cuda.is_available() and args.fp16
    dtype = torch.float16 if use_amp else torch.float32
    
    if use_amp:
        logger.log("DEBUG", "AMP enabled for inference", "⚡")
    
    with torch.autocast(
        device_type='cuda' if torch.cuda.is_available() else 'cpu',
        dtype=dtype,
        enabled=use_amp
    ):
        # Calculate optimal scale factor for this chunk
        scale_factor = calculate_optimal_scale_factor(h, w, logger)
        
        # 1. Compute flows with safe inference
        gt_flows_bi = safe_raft_inference(
            video_tensor, raft_model, scale_factor, args.raft_iter, logger,
            enable_cpu_fallback=not args.no_cpu_fallback
        )
        
        # 2. Complete flows (FP32 for stability)
        logger.start_stage("Flow Completion")
        with torch.no_grad():
            pred_flows_bi, _ = flow_complete_model.forward_bidirect_flow(
                gt_flows_bi, mask_tensor.float()
            )
            pred_flows_bi = flow_complete_model.combine_flow(
                gt_flows_bi, pred_flows_bi, mask_tensor.float()
            )
        logger.end_stage("Flow Completion")
        
        # 3. Temporal propagation
        logger.start_stage("Temporal Propagation")
        with torch.no_grad():
            prop_imgs, updated_local_masks = model.img_propagation(
                video_tensor * (1 - mask_tensor),
                pred_flows_bi,
                mask_tensor,
                'nearest'
            )
        
        updated_masks = updated_local_masks.view(b, t, 1, h, w)
        updated_frames = video_tensor * (1 - mask_tensor) + prop_imgs.view(b, t, c, h, w) * mask_tensor
        logger.end_stage("Temporal Propagation")
        
        # 4. Final inference
        logger.start_stage("Final Inference")
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
        
        logger.end_stage("Final Inference")
        logger.end_stage("Chunk Processing")
        
        return comp_frames_tensor


def main(args):
    """Main inference function with detailed logging and error handling"""
    logger = InferenceLogger(log_level=args.log_level)
    logger.log("INFO", "🚀 Starting ProPainter Inference (Stable v3)", "🚀")
    logger.log("DEBUG", f"Arguments: {vars(args)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log("INFO", f"Using device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.log("INFO", f"GPU: {gpu_name} ({total_memory:.1f} GB)")
    
    # Auto-detect weight paths if not provided
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(base_dir, 'weights')
    
    raft_path = args.raft_model_path if args.raft_model_path else os.path.join(weights_dir, 'raft-things.pth')
    fc_path = args.fc_model_path if args.fc_model_path else os.path.join(weights_dir, 'recurrent_flow_completion.pth')
    propainter_path = args.model_path
    
    logger.log("INFO", f"RAFT weights: {os.path.basename(raft_path)}", "📦")
    logger.log("INFO", f"Flow completion weights: {os.path.basename(fc_path)}", "📦")
    logger.log("INFO", f"ProPainter weights: {os.path.basename(propainter_path)}", "📦")
    
    # 1. Load RAFT (Flow Estimation) - Keep in FP32 for stability
    logger.start_stage("Model Loading")
    if not os.path.exists(raft_path):
        raise FileNotFoundError(f"RAFT weights not found at {raft_path}")
    fix_raft = RAFT_bi(raft_path, device)
    fix_raft.eval()
    
    # 2. Load Flow Completion Model - Keep in FP32
    if not os.path.exists(fc_path):
        raise FileNotFoundError(f"Flow completion weights not found at {fc_path}")
    fix_flow_complete = RecurrentFlowCompleteNet(fc_path).to(device)
    fix_flow_complete.eval()
    
    # 3. Load ProPainter Model
    if not os.path.exists(propainter_path):
        raise FileNotFoundError(f"ProPainter weights not found at {propainter_path}")
    model = InpaintGenerator(model_path=propainter_path).to(device)
    model.eval()
    
    # Enable torch.compile if available (PyTorch 2.x)
    if torch.cuda.is_available() and hasattr(torch, 'compile'):
        try:
            logger.log("DEBUG", "Enabling torch.compile for performance boost", "⚡")
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            logger.log("INFO", "torch.compile enabled", "✅")
        except Exception as e:
            logger.log("WARNING", f"torch.compile failed: {e}", "⚠️")
            logger.log("INFO", "Falling back to eager mode", "⚠️")
    
    logger.end_stage("Model Loading")
    
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
    logger.log("INFO", f"Resolution: {orig_w}x{orig_h} ({orig_w*orig_h:,} pixels)", "📏")
    logger.log("INFO", f"Total frames: {len(frames)}", "📊")

    os.makedirs(args.output, exist_ok=True)

    # 5. Load and preprocess all frames
    logger.start_stage("Data Preprocessing")
    video_data = []
    mask_data = []

    for i, (f_path, m_path) in enumerate(zip(frames, masks)):
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
        
        if (i + 1) % 10 == 0:
            logger.log("DEBUG", f"Preprocessed {i + 1}/{len(frames)} frames")

    # Stack to Tensor: [1, T, C, H, W]
    video_tensor = torch.stack(video_data).unsqueeze(0).to(device)
    mask_tensor = torch.stack(mask_data).unsqueeze(0).to(device)
    
    logger.log("INFO", f"Tensor shape: {video_tensor.shape}")
    logger.log("DEBUG", f"Memory usage: {video_tensor.numel() * video_tensor.element_size() / 1024**3:.2f} GB")
    logger.end_stage("Data Preprocessing")
    
    # 6. Process video (with chunking for long videos)
    T = video_tensor.shape[1]
    if T > 50:  # Use chunking for long videos
        logger.log("INFO", f"Using chunked processing for {T} frames", "🔀")
        comp_frames_tensor = process_video_in_chunks(
            video_tensor, mask_tensor, model, 
            fix_raft, fix_flow_complete,
            args, logger
        )
    else:
        logger.log("INFO", f"Processing all {T} frames at once", "🔀")
        comp_frames_tensor = process_single_chunk(
            video_tensor, mask_tensor, model, fix_raft, fix_flow_complete, args, logger
        )
    
    # 7. Save results
    logger.start_stage("Saving Results")
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
            logger.log("DEBUG", f"Saved {i + 1}/{len(frames)} frames")
    
    logger.end_stage("Saving Results")
    logger.log("INFO", f"Results saved to {args.output}", "✅")
    logger.print_summary()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Optimized ProPainter Inference with Smart Downscale and CPU Fallback'
    )
    
    # Required arguments
    parser.add_argument('--video', type=str, required=True, 
                       help='Path to input frames folder')
    parser.add_argument('--mask', type=str, required=True, 
                       help='Path to input masks folder')
    parser.add_argument('--output', type=str, required=True, 
                       help='Path to output folder')
    
    # Model paths (with auto-detection)
    parser.add_argument('--model_path', type=str, default='weights/ProPainter.pth',
                       help='Path to ProPainter .pth model (default: weights/ProPainter.pth)')
    parser.add_argument('--raft_model_path', type=str, default=None,
                       help='Path to RAFT .pth model (auto-detected if not provided)')
    parser.add_argument('--fc_model_path', type=str, default=None,
                       help='Path to flow completion .pth model (auto-detected if not provided)')
    
    # Inference parameters
    parser.add_argument('--raft_iter', type=int, default=20, 
                       help='RAFT iterations (default: 20)')
    parser.add_argument('--ref_stride', type=int, default=10, 
                       help='Reference frame stride (default: 10)')
    parser.add_argument('--neighbor_length', type=int, default=20, 
                       help='Neighbor window length (default: 20)')
    parser.add_argument('--chunk_size', type=int, default=10, 
                       help='Number of frames to process at once (default: 10)')
    
    # Optimization flags
    parser.add_argument('--fp16', action='store_true', default=True,
                       help='Use Automatic Mixed Precision (AMP) for faster inference (default: True)')
    parser.add_argument('--no-fp16', dest='fp16', action='store_false',
                       help='Disable Automatic Mixed Precision (AMP)')
    parser.add_argument('--no-cpu-fallback', action='store_true', default=False,
                       help='Disable CPU fallback on OOM errors (default: fallback enabled)')
    
    # Logging
    parser.add_argument('--log-level', type=str, default='DEBUG',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: DEBUG)')
    
    args = parser.parse_args()
    main(args)
