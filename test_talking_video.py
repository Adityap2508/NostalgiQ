#!/usr/bin/env python3
"""
Test script for Talking Video Generator

This script tests the basic functionality without requiring
full SadTalker setup.
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing package imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not available")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not available")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow available")
    except ImportError:
        print("❌ Pillow not available")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError:
        print("❌ NumPy not available")
        return False
    
    try:
        from TTS.api import TTS
        print("✅ TTS library available")
    except ImportError:
        print("❌ TTS library not available")
        return False
    
    return True

def test_tts():
    """Test TTS functionality"""
    print("\n🎵 Testing TTS functionality...")
    
    try:
        from TTS.api import TTS
        
        # List available models
        print("Available TTS models:")
        models = TTS.list_models()
        for i, model in enumerate(models[:5]):  # Show first 5
            print(f"  {i+1}. {model}")
        
        # Test loading a model
        print("\nLoading TTS model...")
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
        print("✅ TTS model loaded successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ TTS test failed: {e}")
        return False

def test_sadtalker_setup():
    """Test SadTalker setup"""
    print("\n🤖 Testing SadTalker setup...")
    
    sadtalker_path = Path("SadTalker")
    
    if not sadtalker_path.exists():
        print("❌ SadTalker directory not found")
        print("Run: python setup_talking_video.py")
        return False
    
    # Check required files
    required_files = [
        "inference.py",
        "src",
        "checkpoints"
    ]
    
    for file in required_files:
        if not (sadtalker_path / file).exists():
            print(f"❌ Missing: {file}")
            return False
    
    print("✅ SadTalker setup looks good!")
    return True

def test_generator():
    """Test the talking video generator"""
    print("\n🎬 Testing Talking Video Generator...")
    
    try:
        from talking_video_generator import TalkingVideoGenerator
        
        # Initialize generator
        generator = TalkingVideoGenerator()
        print("✅ Generator initialized successfully!")
        
        # Test TTS model listing
        generator.list_available_tts_models()
        
        return True
        
    except Exception as e:
        print(f"❌ Generator test failed: {e}")
        return False

def create_test_image():
    """Create a simple test image"""
    print("\n🖼️ Creating test image...")
    
    try:
        import cv2
        import numpy as np
        
        # Create a simple test image with a face-like shape
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Draw a simple face
        cv2.ellipse(img, (200, 200), (80, 100), 0, 0, 360, (220, 200, 180), -1)
        cv2.circle(img, (180, 180), 10, (0, 0, 0), -1)  # Left eye
        cv2.circle(img, (220, 180), 10, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(img, (200, 230), (20, 10), 0, 0, 180, (0, 0, 0), 2)  # Mouth
        
        # Save image
        cv2.imwrite("test_face.jpg", img)
        print("✅ Test image created: test_face.jpg")
        
        return True
        
    except Exception as e:
        print(f"❌ Test image creation failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Talking Video Generator Test Suite")
    print("=" * 40)
    
    tests = [
        ("Package Imports", test_imports),
        ("TTS Functionality", test_tts),
        ("SadTalker Setup", test_sadtalker_setup),
        ("Generator Initialization", test_generator),
        ("Test Image Creation", create_test_image)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 TEST SUMMARY")
    print("=" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Add your face images to the current directory")
        print("2. Run: python talking_video_generator.py --image your_image.jpg --text 'Your text'")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        print("You may need to run: python setup_talking_video.py")

if __name__ == "__main__":
    main()
