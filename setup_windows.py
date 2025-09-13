#!/usr/bin/env python3
"""
Windows-compatible setup script for Talking Video Generator

This script handles Windows-specific issues including setuptools problems
and provides alternative installation methods.
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

def fix_setuptools():
    """Fix setuptools issues on Windows"""
    print("🔧 Fixing setuptools issues...")
    
    # Update pip first
    run_command("python -m pip install --upgrade pip", "Upgrading pip")
    
    # Install/upgrade setuptools and wheel
    run_command("pip install --upgrade setuptools wheel", "Upgrading setuptools and wheel")
    
    # Try to install build tools
    run_command("pip install build", "Installing build tools")
    
    return True

def install_python_packages():
    """Install required Python packages with Windows compatibility"""
    # Fix setuptools first
    fix_setuptools()
    
    # Core packages - install one by one for better error handling
    core_packages = [
        "numpy",
        "pillow", 
        "opencv-python",
        "gdown"
    ]
    
    print("Installing core packages...")
    for package in core_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠ Failed to install {package}, trying alternative...")
            # Try with --no-cache-dir
            run_command(f"pip install --no-cache-dir {package}", f"Installing {package} (no cache)")
    
    # Install PyTorch (this can be tricky on Windows)
    print("Installing PyTorch...")
    if not run_command("pip install torch torchvision torchaudio", "Installing PyTorch"):
        print("⚠ PyTorch installation failed, trying CPU-only version...")
        run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu", 
                   "Installing PyTorch CPU version")
    
    # Install TTS packages
    tts_packages = [
        "gtts",  # Google TTS - most reliable
        "pyttsx3"  # Offline TTS
    ]
    
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

def setup_sadtalker_windows():
    """Setup SadTalker with Windows-specific handling"""
    print("🤖 Setting up SadTalker for Windows...")
    
    # Clone repository
    if not os.path.exists("SadTalker"):
        print("Cloning SadTalker repository...")
        if not run_command("git clone https://github.com/OpenTalker/SadTalker.git", 
                          "Cloning SadTalker repository"):
            return False
    else:
        print("✅ SadTalker directory already exists")
    
    # Fix setuptools in SadTalker directory
    print("Fixing setuptools in SadTalker directory...")
    run_command("pip install --upgrade setuptools wheel", "Upgrading setuptools", cwd="SadTalker")
    
    # Install SadTalker requirements with better error handling
    print("Installing SadTalker requirements...")
    
    # Try to install requirements.txt
    if not run_command("pip install -r requirements.txt", 
                      "Installing SadTalker requirements", 
                      cwd="SadTalker"):
        print("⚠ SadTalker requirements installation had issues")
        print("Trying to install key packages individually...")
        
        # Install key packages individually
        key_packages = [
            "face-alignment",
            "imageio",
            "imageio-ffmpeg", 
            "librosa",
            "numba",
            "resampy",
            "scipy",
            "scikit-image",
            "tqdm",
            "yacs"
        ]
        
        for package in key_packages:
            run_command(f"pip install {package}", f"Installing {package}", cwd="SadTalker")
    
    # Download models (this will likely fail on Windows, but we'll try)
    print("Attempting to download SadTalker models...")
    if not run_command("bash scripts/download_models.sh", 
                      "Downloading SadTalker models", 
                      cwd="SadTalker"):
        print("⚠ Model download failed (expected on Windows)")
        print("📝 Note: You may need to download models manually")
        print("   Check the SadTalker GitHub for manual download instructions")
    
    return True

def create_windows_instructions():
    """Create Windows-specific instructions"""
    instructions = """# Windows Setup Instructions

## If you encounter setuptools errors:

1. **Update pip and setuptools:**
   ```bash
   python -m pip install --upgrade pip
   pip install --upgrade setuptools wheel
   ```

2. **Install packages individually:**
   ```bash
   pip install numpy
   pip install pillow
   pip install opencv-python
   pip install gtts
   pip install torch torchvision torchaudio
   ```

3. **For SadTalker models (if download fails):**
   - Visit: https://github.com/OpenTalker/SadTalker
   - Download models manually to SadTalker/checkpoints/
   - Or use the simplified version without SadTalker

## Alternative: Use the Simple Version

If SadTalker setup continues to fail, use the simple version:
```bash
python talking_video_simple.py --image face.jpg --text "Hello!"
```

This version only requires:
- Google TTS (gtts)
- Basic image processing
- No SadTalker dependency
"""
    
    with open("WINDOWS_SETUP.md", "w") as f:
        f.write(instructions)
    
    print("✅ Created WINDOWS_SETUP.md with troubleshooting guide")

def main():
    """Main setup function for Windows"""
    print("🎬 Talking Video Generator - Windows Setup")
    print("=" * 45)
    
    # Check if we're on Windows
    if platform.system() != "Windows":
        print("⚠ This script is optimized for Windows")
        print("You can still run it, but some features may not work as expected")
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install Python packages
    print("\n📦 Installing Python packages...")
    if not install_python_packages():
        print("❌ Package installation failed!")
        print("Check WINDOWS_SETUP.md for troubleshooting")
        return
    
    # Setup SadTalker
    print("\n🤖 Setting up SadTalker...")
    if not setup_sadtalker_windows():
        print("❌ SadTalker setup had issues!")
        print("You can still use the simple version without SadTalker")
    
    # Create instructions
    create_windows_instructions()
    
    print("\n🎉 Windows setup completed!")
    print("\nNext steps:")
    print("1. Try: python talking_video_simple.py --image face.jpg --text 'Hello!'")
    print("2. If SadTalker works: python talking_video_generator.py --image face.jpg --text 'Hello!'")
    print("3. Check WINDOWS_SETUP.md for troubleshooting")

if __name__ == "__main__":
    main()
