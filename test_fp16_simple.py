#!/usr/bin/env python3
"""
Simple test for --fp16 default behavior without importing the full inference_core.py
"""

import argparse
import sys

def create_parser():
    """Create the same parser as in inference_core.py"""
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
    parser.add_argument('--fp16', action='store_true', default=True,
                       help='Use Automatic Mixed Precision (AMP) for faster inference (default: True)')
    parser.add_argument('--no-fp16', dest='fp16', action='store_false',
                       help='Disable Automatic Mixed Precision (AMP)')
    return parser

def test_fp16_default():
    print("🧪 Testing --fp16 default behavior")
    print("=" * 60)
    
    parser = create_parser()
    
    # Test 1: Default behavior (no flag)
    test_args = [
        '--video', 'test_video',
        '--mask', 'test_mask',
        '--output', 'test_output'
    ]
    
    args = parser.parse_args(test_args)
    print(f"Test 1 - Default (no flag):")
    print(f"  --fp16: {args.fp16}")
    
    if args.fp16:
        print("✅ --fp16 is enabled by default (True)")
    else:
        print("❌ --fp16 is not enabled by default")
        return False
    
    # Test 2: Explicitly disable with --no-fp16
    test_args_no_fp16 = [
        '--video', 'test_video',
        '--mask', 'test_mask',
        '--output', 'test_output',
        '--no-fp16'
    ]
    
    args_no_fp16 = parser.parse_args(test_args_no_fp16)
    print(f"\nTest 2 - With --no-fp16:")
    print(f"  --fp16: {args_no_fp16.fp16}")
    
    if not args_no_fp16.fp16:
        print("✅ --no-fp16 correctly disables AMP")
    else:
        print("❌ --no-fp16 does not disable AMP")
        return False
    
    # Test 3: Explicitly enable with --fp16
    test_args_explicit = [
        '--video', 'test_video',
        '--mask', 'test_mask',
        '--output', 'test_output',
        '--fp16'
    ]
    
    args_explicit = parser.parse_args(test_args_explicit)
    print(f"\nTest 3 - With explicit --fp16:")
    print(f"  --fp16: {args_explicit.fp16}")
    
    if args_explicit.fp16:
        print("✅ --fp16 flag works correctly")
    else:
        print("❌ --fp16 flag does not work")
        return False
    
    # Test 4: Check that other arguments work
    test_args_full = [
        '--video', '/path/to/video',
        '--mask', '/path/to/mask',
        '--output', '/path/to/output',
        '--model_path', 'custom_model.pth',
        '--raft_iter', '30',
        '--chunk_size', '20',
        '--fp16'
    ]
    
    args_full = parser.parse_args(test_args_full)
    print(f"\nTest 4 - Full argument set:")
    print(f"  --video: {args_full.video}")
    print(f"  --mask: {args_full.mask}")
    print(f"  --output: {args_full.output}")
    print(f"  --model_path: {args_full.model_path}")
    print(f"  --raft_iter: {args_full.raft_iter}")
    print(f"  --chunk_size: {args_full.chunk_size}")
    print(f"  --fp16: {args_full.fp16}")
    
    if (args_full.video == '/path/to/video' and 
        args_full.mask == '/path/to/mask' and
        args_full.output == '/path/to/output' and
        args_full.model_path == 'custom_model.pth' and
        args_full.raft_iter == 30 and
        args_full.chunk_size == 20 and
        args_full.fp16 == True):
        print("✅ All arguments parsed correctly")
    else:
        print("❌ Some arguments not parsed correctly")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All --fp16 tests passed!")
    print("\n📋 Usage summary:")
    print("  Default: AMP is ENABLED")
    print("  To disable: use --no-fp16 flag")
    print("  To explicitly enable: use --fp16 flag (redundant but works)")
    
    return True

if __name__ == "__main__":
    success = test_fp16_default()
    sys.exit(0 if success else 1)
