#!/usr/bin/env python3
"""
Setup script for Talking Video Generator

This script handles the installation and setup of all dependencies
for the talking video generator.

Run this script once before using the main generator.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(cmd, description, cwd=None):
    """Run a command with error handling"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_python_packages():
    """Install required Python packages"""
    # First, ensure setuptools is up to date
    print("Updating setuptools...")
    run_command("pip install --upgrade setuptools wheel", "Updating setuptools and wheel")
    
    # Core packages
    core_packages = [
        "torch torchvision torchaudio",
        "opencv-python",
        "pillow",
        "numpy",
        "gdown"
    ]
    
    # TTS packages (try multiple options)
    tts_packages = [
        "gtts",  # Google TTS - most reliable
        "pyttsx3",  # Offline TTS
        "TTS"  # Coqui TTS - try last due to version issues
    ]
    
    print("Installing core packages...")
    for package in core_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            return False
    
    print("Installing TTS packages...")
    tts_success = False
    for package in tts_packages:
        if run_command(f"pip install {package}", f"Installing {package}"):
            tts_success = True
            print(f"✓ {package} installed successfully")
        else:
            print(f"⚠ {package} installation failed, trying next option...")
    
    if not tts_success:
        print("❌ No TTS package could be installed!")
        return False
    
    return True

def setup_sadtalker():
    """Setup SadTalker repository and models"""
    # Clone repository
    if not os.path.exists("SadTalker"):
        if not run_command("git clone https://github.com/OpenTalker/SadTalker.git", 
                          "Cloning SadTalker repository"):
            return False
    else:
        print("✅ SadTalker directory already exists")
    
    # Update setuptools in SadTalker directory
    print("Updating setuptools for SadTalker...")
    run_command("pip install --upgrade setuptools wheel", "Updating setuptools", cwd="SadTalker")
    
    # Install SadTalker requirements with error handling
    print("Installing SadTalker requirements...")
    if not run_command("pip install -r requirements.txt", 
                      "Installing SadTalker requirements", 
                      cwd="SadTalker"):
        print("⚠ SadTalker requirements installation had issues, but continuing...")
    
    # Try to download models (this might fail on Windows)
    print("Downloading SadTalker models...")
    if not run_command("bash scripts/download_models.sh", 
                      "Downloading SadTalker models", 
                      cwd="SadTalker"):
        print("⚠ Model download failed (this is common on Windows)")
        print("You may need to download models manually or use a different approach")
    
    return True

def create_example_files():
    """Create example configuration files"""
    
    # Create example batch file
    batch_example = {
        "examples": [
            {
                "image": "person1.jpg",
                "text": "Hello! I'm excited to meet you today."
            },
            {
                "image": "person2.jpg", 
                "text": "This is a demonstration of the talking video generator."
            }
        ]
    }
    
    with open("example_batch.json", "w") as f:
        import json
        json.dump(batch_example, f, indent=2)
    
    print("✅ Created example_batch.json")
    
    # Create README
    readme_content = """# Talking Video Generator

## Quick Start

1. **Setup (run once):**
   ```bash
   python setup_talking_video.py
   ```

2. **Generate a talking video:**
   ```bash
   python talking_video_generator.py --image person.jpg --text "Hello world!"
   ```

3. **Batch processing:**
   ```bash
   python talking_video_generator.py --batch example_batch.json
   ```

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- At least 8GB RAM
- 10GB free disk space

## Input Image Requirements

- Frontal face photo
- Good lighting
- High resolution (256x256 minimum)
- Common formats: JPG, PNG

## Output

- Generated videos saved in `output/` directory
- MP4 format
- Audio synchronized with lip movements

## Troubleshooting

- If TTS fails: Try different TTS models with `--list-models`
- If SadTalker fails: Check GPU memory and try `--still` mode
- For poor quality: Use higher resolution input images
"""
    
    with open("README_talking_video.md", "w") as f:
        f.write(readme_content)
    
    print("✅ Created README_talking_video.md")

def main():
    """Main setup function"""
    print("🎬 Talking Video Generator Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install Python packages
    print("\n📦 Installing Python packages...")
    if not install_python_packages():
        print("❌ Package installation failed!")
        return
    
    # Setup SadTalker
    print("\n🤖 Setting up SadTalker...")
    if not setup_sadtalker():
        print("❌ SadTalker setup failed!")
        return
    
    # Create example files
    print("\n📝 Creating example files...")
    create_example_files()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Add your face images to the current directory")
    print("2. Run: python talking_video_generator.py --image your_image.jpg --text 'Your text here'")
    print("3. Check the 'output' directory for your generated video!")

if __name__ == "__main__":
    main()
