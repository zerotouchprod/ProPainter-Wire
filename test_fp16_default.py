#!/usr/bin/env python3
"""
Test that --fp16 is enabled by default in inference_core.py
"""

import argparse
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_fp16_default():
    print("🧪 Testing --fp16 default behavior")
    print("=" * 60)
    
    # Simulate command line arguments
    test_args = [
        '--video', 'test_video',
        '--mask', 'test_mask',
        '--output', 'test_output'
    ]
    
    # Import the argument parser from inference_core
    from inference_core import parser
    
    # Parse arguments
    args = parser.parse_args(test_args)
    
    print(f"Parsed arguments:")
    print(f"  --video: {args.video}")
    print(f"  --mask: {args.mask}")
    print(f"  --output: {args.output}")
    print(f"  --fp16: {args.fp16}")
    
    # Test default value
    if args.fp16:
        print("✅ --fp16 is enabled by default (True)")
    else:
        print("❌ --fp16 is not enabled by default")
        return False
    
    # Test disabling with --no-fp16
    test_args_no_fp16 = [
        '--video', 'test_video',
        '--mask', 'test_mask',
        '--output', 'test_output',
        '--no-fp16'
    ]
    
    args_no_fp16 = parser.parse_args(test_args_no_fp16)
    print(f"\nWith --no-fp16 flag:")
    print(f"  --fp16: {args_no_fp16.fp16}")
    
    if not args_no_fp16.fp16:
        print("✅ --no-fp16 correctly disables AMP")
    else:
        print("❌ --no-fp16 does not disable AMP")
        return False
    
    # Test explicitly enabling with --fp16
    test_args_explicit = [
        '--video', 'test_video',
        '--mask', 'test_mask',
        '--output', 'test_output',
        '--fp16'
    ]
    
    args_explicit = parser.parse_args(test_args_explicit)
    print(f"\nWith explicit --fp16 flag:")
    print(f"  --fp16: {args_explicit.fp16}")
    
    if args_explicit.fp16:
        print("✅ --fp16 flag works correctly")
    else:
        print("❌ --fp16 flag does not work")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All --fp16 tests passed!")
    print("\nUsage examples:")
    print("  python inference_core.py --video ... --mask ... --output ...")
    print("    → AMP enabled by default")
    print("  python inference_core.py --video ... --mask ... --output ... --no-fp16")
    print("    → AMP disabled")
    print("  python inference_core.py --video ... --mask ... --output ... --fp16")
    print("    → AMP explicitly enabled")
    
    return True

if __name__ == "__main__":
    success = test_fp16_default()
    sys.exit(0 if success else 1)
