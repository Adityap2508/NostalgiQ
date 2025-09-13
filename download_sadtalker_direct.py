#!/usr/bin/env python3
"""
Direct SadTalker Model Downloader

Downloads SadTalker models directly using Python requests
without requiring gdown.
"""

import os
import requests
from pathlib import Path
import time

def download_file_direct(url, filename, description=""):
    """Download a file directly using requests"""
    print(f"📥 {description}...")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Download with progress
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
        
        print(f"\n✅ Downloaded: {os.path.basename(filename)}")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to download {os.path.basename(filename)}: {e}")
        return False

def download_sadtalker_models():
    """Download SadTalker models from HuggingFace"""
    print("🤖 Downloading SadTalker models from HuggingFace...")
    
    # Create checkpoints directory
    checkpoints_dir = Path("SadTalker/checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Model URLs from HuggingFace (official SadTalker repository)
    models = {
        "SadTalker_V0.0.2_256.safetensors": "https://huggingface.co/vinthony/SadTalker/resolve/main/checkpoints/SadTalker_V0.0.2_256.safetensors",
        "SadTalker_V0.0.2_512.safetensors": "https://huggingface.co/vinthony/SadTalker/resolve/main/checkpoints/SadTalker_V0.0.2_512.safetensors",
        "epoch_20.pth": "https://huggingface.co/vinthony/SadTalker/resolve/main/checkpoints/epoch_20.pth",
        "GFPGANv1.3.pth": "https://huggingface.co/vinthony/SadTalker/resolve/main/checkpoints/GFPGANv1.3.pth",
        "auido2pose_00300-model.pth": "https://huggingface.co/vinthony/SadTalker/resolve/main/checkpoints/auido2pose_00300-model.pth",
        "facevid2vid_00189-model.pth.tar": "https://huggingface.co/vinthony/SadTalker/resolve/main/checkpoints/facevid2vid_00189-model.pth.tar"
    }
    
    print(f"📦 Downloading {len(models)} model files...")
    print("⚠️  Note: These are large files (several GB), download may take time...")
    
    success_count = 0
    for model_name, url in models.items():
        model_path = checkpoints_dir / model_name
        
        if model_path.exists():
            print(f"✓ {model_name} already exists, skipping...")
            success_count += 1
            continue
        
        if download_file_direct(url, str(model_path), f"Downloading {model_name}"):
            success_count += 1
        
        # Small delay between downloads
        time.sleep(1)
    
    print(f"\n📊 Downloaded {success_count}/{len(models)} models")
    
    if success_count >= 2:
        print("✅ SadTalker models ready!")
        return True
    else:
        print("❌ Insufficient models downloaded")
        return False

def test_sadtalker_setup():
    """Test if SadTalker is properly set up"""
    print("\n🧪 Testing SadTalker setup...")
    
    sadtalker_dir = Path("SadTalker")
    if not sadtalker_dir.exists():
        print("❌ SadTalker directory not found")
        return False
    
    checkpoints_dir = sadtalker_dir / "checkpoints"
    if not checkpoints_dir.exists():
        print("❌ Checkpoints directory not found")
        return False
    
    # Check for essential model files
    essential_models = [
        "SadTalker_V0.0.2_256.safetensors",
        "epoch_20.pth"
    ]
    
    found_models = 0
    for model in essential_models:
        if (checkpoints_dir / model).exists():
            found_models += 1
            print(f"✓ Found: {model}")
        else:
            print(f"❌ Missing: {model}")
    
    if found_models >= 1:
        print("✅ SadTalker appears to be ready!")
        return True
    else:
        print("❌ SadTalker setup incomplete")
        return False

def main():
    """Main function"""
    print("🤖 Direct SadTalker Model Downloader")
    print("=" * 40)
    
    # Download models
    if download_sadtalker_models():
        # Test setup
        if test_sadtalker_setup():
            print("\n🎉 SadTalker is ready to use!")
            print("\nYou can now run:")
            print("python sadtalker_working.py --image Photo1.jpg --text 'Trial 1'")
        else:
            print("\n⚠️ SadTalker setup may be incomplete")
    else:
        print("\n❌ SadTalker setup failed")
        print("\n💡 Alternative: Use the simple video generator:")
        print("python simple_video.py")

if __name__ == "__main__":
    main()
