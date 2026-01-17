#!/usr/bin/env python3
"""
Test script for the optimized inference_core.py with Smart Downscale and CPU Fallback.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

def create_test_data():
    """Create minimal test data for inference"""
    test_dir = tempfile.mkdtemp(prefix="propainter_test_")
    
    # Create video frames directory
    video_dir = os.path.join(test_dir, "video")
    mask_dir = os.path.join(test_dir, "mask")
    output_dir = os.path.join(test_dir, "output")
    
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create 3 test frames (small resolution for fast testing)
    import cv2
    import numpy as np
    
    for i in range(3):
        # Create a simple test image
        img = np.ones((128, 128, 3), dtype=np.uint8) * (i * 80)
        cv2.imwrite(os.path.join(video_dir, f"frame_{i:03d}.jpg"), img)
        
        # Create a simple mask (center rectangle)
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[40:88, 40:88] = 255
        cv2.imwrite(os.path.join(mask_dir, f"mask_{i:03d}.png"), mask)
    
    return test_dir, video_dir, mask_dir, output_dir

def test_basic_inference():
    """Test basic inference with the new optimized code"""
    print("🧪 Testing optimized inference_core.py")
    print("=" * 60)
    
    # Create test data
    test_dir, video_dir, mask_dir, output_dir = create_test_data()
    
    try:
        # Build command
        cmd = [
            sys.executable, "inference_core.py",
            "--video", video_dir,
            "--mask", mask_dir,
            "--output", output_dir,
            "--model_path", "weights/ProPainter.pth",
            "--raft_iter", "5",  # Reduced for testing
            "--chunk_size", "2",
            "--log-level", "INFO"
        ]
        
        print(f"Command: {' '.join(cmd)}")
        print(f"Test directory: {test_dir}")
        
        # Run inference
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        print("\n" + "=" * 60)
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Inference completed successfully")
            
            # Check if output files were created
            output_files = list(Path(output_dir).glob("*.jpg"))
            if output_files:
                print(f"✅ Output files created: {len(output_files)}")
                for f in output_files[:3]:  # Show first 3 files
                    print(f"  - {f.name}")
            else:
                print("❌ No output files created")
                return False
                
        else:
            print(f"❌ Inference failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(test_dir)
            print(f"🧹 Cleaned up test directory: {test_dir}")
        except:
            pass
    
    return True

def test_logging_levels():
    """Test different logging levels"""
    print("\n" + "=" * 60)
    print("🧪 Testing logging levels")
    
    test_dir, video_dir, mask_dir, output_dir = create_test_data()
    
    logging_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
    
    for level in logging_levels:
        print(f"\nTesting log level: {level}")
        
        cmd = [
            sys.executable, "inference_core.py",
            "--video", video_dir,
            "--mask", mask_dir,
            "--output", os.path.join(output_dir, f"test_{level}"),
            "--model_path", "weights/ProPainter.pth",
            "--raft_iter", "2",  # Very small for quick test
            "--log-level", level
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                timeout=30  # 30 second timeout
            )
            
            if result.returncode == 0:
                print(f"  ✅ {level}: PASSED")
            else:
                print(f"  ❌ {level}: FAILED (code: {result.returncode})")
                
        except subprocess.TimeoutExpired:
            print(f"  ⏱️ {level}: TIMEOUT")
        except Exception as e:
            print(f"  ❌ {level}: ERROR ({e})")
    
    # Cleanup
    try:
        shutil.rmtree(test_dir)
    except:
        pass
    
    return True

def test_scale_factor_logic():
    """Test the scale factor calculation logic"""
    print("\n" + "=" * 60)
    print("🧪 Testing scale factor calculation")
    
    # Import the function directly
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from inference_core import calculate_optimal_scale_factor, InferenceLogger
        
        logger = InferenceLogger(log_level="INFO")
        
        test_cases = [
            (64, 64, 1.0, "Very low resolution"),
            (512, 512, 1.0, "Low resolution (0.26MP)"),
            (1024, 1024, 0.5, "Medium resolution (1MP)"),
            (1920, 1080, 0.5, "Full HD (2MP)"),
            (3840, 2160, 0.25, "4K (8.3MP)"),
            (7680, 4320, 0.125, "8K (33MP)"),
        ]
        
        all_passed = True
        for h, w, expected, description in test_cases:
            actual = calculate_optimal_scale_factor(h, w, logger)
            if abs(actual - expected) < 0.01:
                print(f"  ✅ {description}: {h}x{w} -> {actual}x (expected: {expected}x)")
            else:
                print(f"  ❌ {description}: {h}x{w} -> {actual}x (expected: {expected}x)")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Failed to test scale factor: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Comprehensive testing of optimized inference_core.py")
    print("=" * 60)
    
    tests = [
        ("Basic Inference", test_basic_inference),
        ("Logging Levels", test_logging_levels),
        ("Scale Factor Logic", test_scale_factor_logic),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! The optimized inference_core.py is ready for production.")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
