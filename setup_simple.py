#!/usr/bin/env python3
"""
Simple setup script for Talking Video Generator (Simplified Version)

This script installs the minimal dependencies needed for the simplified version
that uses Google TTS instead of Coqui TTS.
"""

import os
import sys
import subprocess
import platform

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
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_python_packages():
    """Install required Python packages"""
    packages = [
        "gtts",  # Google TTS - most reliable
        "torch torchvision torchaudio",
        "opencv-python",
        "pillow",
        "numpy",
        "gdown"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
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
    
    # Install SadTalker requirements
    if not run_command("pip install -r requirements.txt", 
                      "Installing SadTalker requirements", 
                      cwd="SadTalker"):
        return False
    
    # Download models
    if not run_command("bash scripts/download_models.sh", 
                      "Downloading SadTalker models", 
                      cwd="SadTalker"):
        return False
    
    return True

def create_example_files():
    """Create example configuration files"""
    
    # Create example batch file
    batch_example = [
        {
            "image": "person1.jpg",
            "text": "Hello! I'm excited to meet you today."
        },
        {
            "image": "person2.jpg", 
            "text": "This is a demonstration of the talking video generator."
        }
    ]
    
    with open("example_batch_simple.json", "w") as f:
        import json
        json.dump(batch_example, f, indent=2)
    
    print("✅ Created example_batch_simple.json")
    
    # Create README
    readme_content = """# Simple Talking Video Generator

## Quick Start

1. **Setup (run once):**
   ```bash
   python setup_simple.py
   ```

2. **Generate a talking video:**
   ```bash
   python talking_video_simple.py --image person.jpg --text "Hello world!"
   ```

3. **Batch processing:**
   ```bash
   python talking_video_simple.py --batch example_batch_simple.json
   ```

## Requirements

- Python 3.7+
- CUDA-compatible GPU (recommended)
- At least 4GB RAM
- 5GB free disk space

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

- If TTS fails: Check internet connection (Google TTS requires internet)
- If SadTalker fails: Check GPU memory and try `--still` mode
- For poor quality: Use higher resolution input images
"""
    
    with open("README_simple.md", "w") as f:
        f.write(readme_content)
    
    print("✅ Created README_simple.md")

def main():
    """Main setup function"""
    print("🎬 Simple Talking Video Generator Setup")
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
    print("2. Run: python talking_video_simple.py --image your_image.jpg --text 'Your text here'")
    print("3. Check the 'output' directory for your generated video!")

if __name__ == "__main__":
    main()
