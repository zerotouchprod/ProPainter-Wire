"""
Unit tests for optimized sparse transformer implementation.
Tests correctness and performance improvements.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.modules.sparse_transformer import SparseWindowAttention as OriginalAttention
from model.modules.sparse_transformer_simple_optimized import SimpleOptimizedSparseWindowAttention as OptimizedAttention


def test_attention_shapes():
    """Test that both attention implementations produce same output shapes."""
    print("Testing attention output shapes...")
    
    # Test parameters
    batch_size = 2
    time_steps = 5
    height, width = 64, 64
    channels = 512
    num_heads = 8
    window_size = (8, 8)
    
    # Create input tensor
    x = torch.randn(batch_size, time_steps, height, width, channels)
    mask = torch.randn(batch_size, time_steps, height, width, 1) > 0
    
    # Initialize attention modules
    original_attn = OriginalAttention(
        dim=channels,
        n_head=num_heads,
        window_size=window_size,
        pooling_token=False
    )
    
    optimized_attn = OptimizedAttention(
        dim=channels,
        n_head=num_heads,
        window_size=window_size,
        pooling_token=False
    )
    
    # Copy weights from original to optimized
    optimized_attn.load_state_dict(original_attn.state_dict())
    
    # Set to eval mode
    original_attn.eval()
    optimized_attn.eval()
    
    # Forward pass
    with torch.no_grad():
        out_original = original_attn(x, mask)
        out_optimized = optimized_attn(x, mask)
    
    # Check shapes
    assert out_original.shape == out_optimized.shape, \
        f"Shape mismatch: original {out_original.shape}, optimized {out_optimized.shape}"
    
    print(f"✅ Shapes match: {out_original.shape}")
    return True


def test_attention_numerical_accuracy(rtol=1e-3, atol=1e-5):
    """Test numerical accuracy between original and optimized implementations."""
    print("\nTesting numerical accuracy...")
    
    # Smaller test for numerical stability
    batch_size = 1
    time_steps = 3
    height, width = 32, 32
    channels = 128
    num_heads = 4
    window_size = (8, 8)
    
    # Create input tensor
    torch.manual_seed(42)
    x = torch.randn(batch_size, time_steps, height, width, channels)
    mask = (torch.randn(batch_size, time_steps, height, width, 1) > 0).float()
    
    # Initialize attention modules
    original_attn = OriginalAttention(
        dim=channels,
        n_head=num_heads,
        window_size=window_size,
        pooling_token=False,
        attn_drop=0.0,
        proj_drop=0.0
    )
    
    optimized_attn = OptimizedAttention(
        dim=channels,
        n_head=num_heads,
        window_size=window_size,
        pooling_token=False,
        attn_drop=0.0,
        proj_drop=0.0
    )
    
    # Copy weights exactly
    optimized_attn.load_state_dict(original_attn.state_dict())
    
    # Set to eval mode
    original_attn.eval()
    optimized_attn.eval()
    
    # Forward pass
    with torch.no_grad():
        out_original = original_attn(x, mask)
        out_optimized = optimized_attn(x, mask)
    
    # Calculate differences
    diff = torch.abs(out_original - out_optimized)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    
    # Check if outputs are close
    is_close = torch.allclose(out_original, out_optimized, rtol=rtol, atol=atol)
    
    if is_close:
        print(f"✅ Numerical accuracy within tolerance (rtol={rtol}, atol={atol})")
    else:
        print(f"⚠️ Numerical differences exceed tolerance")
        
        # Print some sample values for debugging
        print("\nSample values (first 5 elements):")
        print("Original:", out_original.flatten()[:5].cpu().numpy())
        print("Optimized:", out_optimized.flatten()[:5].cpu().numpy())
    
    return is_close


def test_memory_usage():
    """Test memory usage comparison."""
    print("\nTesting memory usage...")
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping memory test")
        return True
    
    # Larger test for memory measurement
    batch_size = 2
    time_steps = 10
    height, width = 128, 128
    channels = 256
    num_heads = 8
    window_size = (8, 8)
    
    # Create input tensor on GPU
    x = torch.randn(batch_size, time_steps, height, width, channels).cuda()
    mask = (torch.randn(batch_size, time_steps, height, width, 1) > 0).float().cuda()
    
    # Initialize attention modules on GPU
    original_attn = OriginalAttention(
        dim=channels,
        n_head=num_heads,
        window_size=window_size,
        pooling_token=False
    ).cuda()
    
    optimized_attn = OptimizedAttention(
        dim=channels,
        n_head=num_heads,
        window_size=window_size,
        pooling_token=False
    ).cuda()
    
    # Copy weights
    optimized_attn.load_state_dict(original_attn.state_dict())
    
    # Set to eval mode
    original_attn.eval()
    optimized_attn.eval()
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Measure memory for original
    with torch.no_grad():
        out_original = original_attn(x, mask)
    
    mem_original = torch.cuda.max_memory_allocated()
    print(f"Original memory: {mem_original / 1024**2:.2f} MB")
    
    # Clear cache
    del out_original
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Measure memory for optimized
    with torch.no_grad():
        out_optimized = optimized_attn(x, mask)
    
    mem_optimized = torch.cuda.max_memory_allocated()
    print(f"Optimized memory: {mem_optimized / 1024**2:.2f} MB")
    
    # Calculate improvement
    if mem_original > 0:
        improvement = (mem_original - mem_optimized) / mem_original * 100
        print(f"Memory improvement: {improvement:.1f}%")
    
    # Clean up
    del out_optimized
    torch.cuda.empty_cache()
    
    return True


def test_performance():
    """Test performance comparison."""
    print("\nTesting performance...")
    
    # Medium test for performance measurement
    batch_size = 1
    time_steps = 5
    height, width = 96, 96
    channels = 192
    num_heads = 6
    window_size = (8, 8)
    
    # Create input tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(batch_size, time_steps, height, width, channels).to(device)
    mask = (torch.randn(batch_size, time_steps, height, width, 1) > 0).float().to(device)
    
    # Initialize attention modules
    original_attn = OriginalAttention(
        dim=channels,
        n_head=num_heads,
        window_size=window_size,
        pooling_token=False
    ).to(device)
    
    optimized_attn = OptimizedAttention(
        dim=channels,
        n_head=num_heads,
        window_size=window_size,
        pooling_token=False
    ).to(device)
    
    # Copy weights
    optimized_attn.load_state_dict(original_attn.state_dict())
    
    # Set to eval mode
    original_attn.eval()
    optimized_attn.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = original_attn(x, mask)
            _ = optimized_attn(x, mask)
    
    # Benchmark original
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    
    with torch.no_grad():
        for _ in range(10):
            _ = original_attn(x, mask)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_original = (time.time() - start) / 10
    
    # Benchmark optimized
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    
    with torch.no_grad():
        for _ in range(10):
            _ = optimized_attn(x, mask)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_optimized = (time.time() - start) / 10
    
    print(f"Original time: {time_original:.4f}s per forward pass")
    print(f"Optimized time: {time_optimized:.4f}s per forward pass")
    
    if time_original > 0:
        speedup = time_original / time_optimized
        print(f"Speedup: {speedup:.2f}x")
    
    return True


def test_fp16_compatibility():
    """Test FP16 compatibility."""
    print("\nTesting FP16 compatibility...")
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping FP16 test")
        return True
    
    # Test parameters
    batch_size = 1
    time_steps = 3
    height, width = 32, 32
    channels = 128
    num_heads = 4
    window_size = (8, 8)
    
    # Create input tensor in FP16
    x = torch.randn(batch_size, time_steps, height, width, channels).half().cuda()
    mask = (torch.randn(batch_size, time_steps, height, width, 1) > 0).float().half().cuda()
    
    # Initialize attention modules
    original_attn = OriginalAttention(
        dim=channels,
        n_head=num_heads,
        window_size=window_size,
        pooling_token=False
    ).cuda()
    
    optimized_attn = OptimizedAttention(
        dim=channels,
        n_head=num_heads,
        window_size=window_size,
        pooling_token=False
    ).cuda()
    
    # Convert to FP16
    original_attn.half()
    optimized_attn.half()
    
    # Copy weights
    optimized_attn.load_state_dict(original_attn.state_dict())
    
    # Set to eval mode
    original_attn.eval()
    optimized_attn.eval()
    
    try:
        # Forward pass in FP16
        with torch.no_grad():
            out_original = original_attn(x, mask)
            out_optimized = optimized_attn(x, mask)
        
        print("✅ FP16 forward pass successful")
        
        # Check for NaN or Inf
        if torch.any(torch.isnan(out_original)) or torch.any(torch.isinf(out_original)):
            print("⚠️ Original output contains NaN/Inf in FP16")
        
        if torch.any(torch.isnan(out_optimized)) or torch.any(torch.isinf(out_optimized)):
            print("⚠️ Optimized output contains NaN/Inf in FP16")
        
        return True
        
    except Exception as e:
        print(f"❌ FP16 test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Optimized Sparse Window Attention")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    tests = [
        test_attention_shapes,
        test_attention_numerical_accuracy,
        test_memory_usage,
        test_performance,
        test_fp16_compatibility,
    ]
    
    for test in tests:
        try:
            if not test():
                all_passed = False
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            print(f"❌ {test.__name__} raised exception: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("⚠️ Some tests failed")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
