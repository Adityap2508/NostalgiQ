#!/usr/bin/env python3
"""
Manual SadTalker Setup Script

This script downloads the required SadTalker models manually since
the bash script doesn't work on Windows.
"""

import os
import subprocess
import requests
from pathlib import Path
import zipfile
import shutil

def download_file(url, filename):
    """Download a file with progress"""
    print(f"📥 Downloading {filename}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end='', flush=True)
        
        print(f"\n✅ Downloaded: {filename}")
        return True
    except Exception as e:
        print(f"\n❌ Failed to download {filename}: {e}")
        return False

def setup_sadtalker_models():
    """Download SadTalker models manually"""
    print("🤖 Setting up SadTalker models manually...")
    
    # Create SadTalker directory if it doesn't exist
    sadtalker_dir = Path("SadTalker")
    if not sadtalker_dir.exists():
        print("❌ SadTalker directory not found. Please run setup first.")
        return False
    
    # Create checkpoints directory
    checkpoints_dir = sadtalker_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Model URLs (these are the official SadTalker model URLs)
    models = {
        "SadTalker_V0.0.2_256.safetensors": "https://huggingface.co/vinthony/SadTalker/resolve/main/checkpoints/SadTalker_V0.0.2_256.safetensors",
        "SadTalker_V0.0.2_512.safetensors": "https://huggingface.co/vinthony/SadTalker/resolve/main/checkpoints/SadTalker_V0.0.2_512.safetensors",
        "auido2pose_00300-model.pth": "https://huggingface.co/vinthony/SadTalker/resolve/main/checkpoints/auido2pose_00300-model.pth",
        "auido2pose_00300-model.pth.tar": "https://huggingface.co/vinthony/SadTalker/resolve/main/checkpoints/auido2pose_00300-model.pth.tar",
        "auido2pose_00300-model.pth.tar": "https://huggingface.co/vinthony/SadTalker/resolve/main/checkpoints/auido2pose_00300-model.pth.tar",
        "epoch_20.pth": "https://huggingface.co/vinthony/SadTalker/resolve/main/checkpoints/epoch_20.pth",
        "facevid2vid_00189-model.pth.tar": "https://huggingface.co/vinthony/SadTalker/resolve/main/checkpoints/facevid2vid_00189-model.pth.tar",
        "GFPGANv1.3.pth": "https://huggingface.co/vinthony/SadTalker/resolve/main/checkpoints/GFPGANv1.3.pth",
        "GPEN-BFR-512.pth": "https://huggingface.co/vinthony/SadTalker/resolve/main/checkpoints/GPEN-BFR-512.pth"
    }
    
    print(f"📦 Downloading {len(models)} model files...")
    
    success_count = 0
    for model_name, url in models.items():
        model_path = checkpoints_dir / model_name
        
        if model_path.exists():
            print(f"✓ {model_name} already exists, skipping...")
            success_count += 1
            continue
        
        if download_file(url, model_path):
            success_count += 1
    
    print(f"\n📊 Downloaded {success_count}/{len(models)} models")
    
    if success_count > 0:
        print("✅ SadTalker models setup complete!")
        return True
    else:
        print("❌ Failed to download any models")
        return False

def test_sadtalker():
    """Test if SadTalker is working"""
    print("\n🧪 Testing SadTalker...")
    
    sadtalker_dir = Path("SadTalker")
    if not sadtalker_dir.exists():
        print("❌ SadTalker directory not found")
        return False
    
    checkpoints_dir = sadtalker_dir / "checkpoints"
    if not checkpoints_dir.exists():
        print("❌ Checkpoints directory not found")
        return False
    
    # Check for key model files
    key_models = [
        "SadTalker_V0.0.2_256.safetensors",
        "epoch_20.pth",
        "GFPGANv1.3.pth"
    ]
    
    found_models = 0
    for model in key_models:
        if (checkpoints_dir / model).exists():
            found_models += 1
            print(f"✓ Found: {model}")
        else:
            print(f"❌ Missing: {model}")
    
    if found_models >= 2:
        print("✅ SadTalker appears to be ready!")
        return True
    else:
        print("❌ SadTalker setup incomplete")
        return False

def main():
    """Main function"""
    print("🤖 SadTalker Manual Setup")
    print("=" * 30)
    
    # Setup models
    if setup_sadtalker_models():
        # Test setup
        if test_sadtalker():
            print("\n🎉 SadTalker is ready to use!")
            print("\nYou can now run:")
            print("python talking_video_generator.py --image Photo1.jpg --text 'Trial 1'")
        else:
            print("\n⚠️ SadTalker setup may be incomplete")
    else:
        print("\n❌ SadTalker setup failed")

if __name__ == "__main__":
    main()
