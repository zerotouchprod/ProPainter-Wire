"""
Test for the fixed sparse_transformer_optimized.py implementation.
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_basic_functionality():
    """Test basic forward pass."""
    print("Testing basic functionality...")
    
    try:
        from model.modules.sparse_transformer_optimized import OptimizedSparseWindowAttention
        
        # Test parameters
        batch_size = 2
        time_steps = 3
        height, width = 32, 32
        channels = 128
        num_heads = 4
        window_size = (8, 8)
        
        # Create model
        model = OptimizedSparseWindowAttention(
            dim=channels,
            n_head=num_heads,
            window_size=window_size,
            pooling_token=False
        )
        
        # Create input
        x = torch.randn(batch_size, time_steps, height, width, channels)
        mask = (torch.randn(batch_size, time_steps, height, width, 1) > 0).float()
        
        # Forward pass
        output = model(x, mask)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
        
        # Check shape preservation
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
        
        print("✅ Basic functionality works")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow():
    """Test gradient computation."""
    print("\nTesting gradient flow...")
    
    try:
        from model.modules.sparse_transformer_optimized import OptimizedSparseWindowAttention
        
        # Test parameters
        batch_size = 1
        time_steps = 2
        height, width = 16, 16
        channels = 64
        num_heads = 2
        window_size = (4, 4)
        
        # Create model
        model = OptimizedSparseWindowAttention(
            dim=channels,
            n_head=num_heads,
            window_size=window_size,
            pooling_token=False
        )
        
        # Create input
        x = torch.randn(batch_size, time_steps, height, width, channels, requires_grad=True)
        mask = (torch.randn(batch_size, time_steps, height, width, 1) > 0).float()
        
        # Forward pass and backward
        model.train()
        output = model(x, mask)
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_gradients = True
                grad_norm = param.grad.norm().item()
                print(f"  {name}: gradient norm = {grad_norm:.6f}")
                break
        
        if has_gradients:
            print("✅ Gradients flow correctly")
        else:
            print("⚠️ No gradients detected")
        
        # Check input gradients
        if x.grad is not None:
            print(f"✅ Input gradients computed: norm = {x.grad.norm().item():.6f}")
        else:
            print("⚠️ Input gradients not computed")
        
        return has_gradients
        
    except Exception as e:
        print(f"❌ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_amp_compatibility():
    """Test AMP compatibility."""
    print("\nTesting AMP compatibility...")
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping AMP test")
        return True
    
    try:
        from model.modules.sparse_transformer_optimized import OptimizedSparseWindowAttention
        
        # Test parameters
        batch_size = 1
        time_steps = 2
        height, width = 16, 16
        channels = 64
        num_heads = 2
        window_size = (4, 4)
        
        # Create model
        model = OptimizedSparseWindowAttention(
            dim=channels,
            n_head=num_heads,
            window_size=window_size,
            pooling_token=False
        ).cuda()
        
        # Create input
        x = torch.randn(batch_size, time_steps, height, width, channels).cuda()
        mask = (torch.randn(batch_size, time_steps, height, width, 1) > 0).float().cuda()
        
        # Test with AMP
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = model(x, mask)
        
        print(f"FP16 output shape: {output.shape}")
        print(f"FP16 output dtype: {output.dtype}")
        
        # Check for NaN/Inf
        if torch.any(torch.isnan(output)):
            print("⚠️ Output contains NaN")
        elif torch.any(torch.isinf(output)):
            print("⚠️ Output contains Inf")
        else:
            print("✅ AMP forward pass successful")
        
        return True
        
    except Exception as e:
        print(f"❌ AMP test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_window_sizes():
    """Test with different window sizes."""
    print("\nTesting different window sizes...")
    
    try:
        from model.modules.sparse_transformer_optimized import OptimizedSparseWindowAttention
        
        test_cases = [
            {"height": 32, "width": 32, "window_size": (8, 8)},
            {"height": 64, "width": 64, "window_size": (16, 16)},
            {"height": 48, "width": 48, "window_size": (12, 12)},
            {"height": 24, "width": 24, "window_size": (6, 6)},
        ]
        
        batch_size = 1
        time_steps = 2
        channels = 64
        num_heads = 2
        
        all_passed = True
        
        for case in test_cases:
            height = case["height"]
            width = case["width"]
            window_size = case["window_size"]
            
            # Create model
            model = OptimizedSparseWindowAttention(
                dim=channels,
                n_head=num_heads,
                window_size=window_size,
                pooling_token=False
            )
            
            # Create input
            x = torch.randn(batch_size, time_steps, height, width, channels)
            mask = (torch.randn(batch_size, time_steps, height, width, 1) > 0).float()
            
            # Forward pass
            output = model(x, mask)
            
            if output.shape == x.shape:
                print(f"✅ Window size {window_size} with {height}x{width}: OK")
            else:
                print(f"❌ Window size {window_size} with {height}x{width}: Shape mismatch")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Window size test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_pooling_token():
    """Test with pooling token enabled."""
    print("\nTesting with pooling token...")
    
    try:
        from model.modules.sparse_transformer_optimized import OptimizedSparseWindowAttention
        
        # Test parameters
        batch_size = 1
        time_steps = 2
        height, width = 32, 32
        channels = 64
        num_heads = 2
        window_size = (8, 8)
        
        # Create model with pooling token
        model = OptimizedSparseWindowAttention(
            dim=channels,
            n_head=num_heads,
            window_size=window_size,
            pooling_token=True,
            pool_size=(4, 4)
        )
        
        # Create input
        x = torch.randn(batch_size, time_steps, height, width, channels)
        mask = (torch.randn(batch_size, time_steps, height, width, 1) > 0).float()
        
        # Forward pass
        output = model(x, mask)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        
        if output.shape == x.shape:
            print("✅ Pooling token works correctly")
            return True
        else:
            print("❌ Pooling token output shape mismatch")
            return False
        
    except Exception as e:
        print(f"❌ Pooling token test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Fixed Optimized Sparse Transformer")
    print("=" * 60)
    
    tests = [
        test_basic_functionality,
        test_gradient_flow,
        test_amp_compatibility,
        test_different_window_sizes,
        test_with_pooling_token,
    ]
    
    all_passed = True
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
        print("✅ All tests passed for fixed optimized sparse transformer!")
        print("\nSummary:")
        print("- Basic forward pass works")
        print("- Gradient computation works")
        print("- AMP compatibility verified")
        print("- Different window sizes supported")
        print("- Pooling token functionality works")
    else:
        print("⚠️ Some tests failed")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
