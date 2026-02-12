#!/usr/bin/env python3
"""
Quick fix for setuptools issues

Run this script to fix the setuptools.build_meta error
"""

import subprocess
import sys

def run_command(cmd, description):
    """Run a command with error handling"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Fix setuptools issues"""
    print("🔧 Fixing setuptools issues...")
    print("=" * 30)
    
    # Step 1: Update pip
    run_command("python -m pip install --upgrade pip", "Upgrading pip")
    
    # Step 2: Install/upgrade setuptools and wheel
    run_command("pip install --upgrade setuptools wheel", "Upgrading setuptools and wheel")
    
    # Step 3: Install build tools
    run_command("pip install build", "Installing build tools")
    
    # Step 4: Try to install a test package
    print("\n🧪 Testing installation...")
    if run_command("pip install gtts", "Testing package installation"):
        print("✅ setuptools fix successful!")
        print("\nYou can now run:")
        print("python setup_talking_video.py")
    else:
        print("❌ setuptools fix failed")
        print("\nTry running:")
        print("python setup_windows.py")
        print("\nOr use the simple version:")
        print("pip install gtts torch torchvision torchaudio opencv-python pillow numpy")

if __name__ == "__main__":
    main()
