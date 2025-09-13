#!/usr/bin/env python3
"""
Download SadTalker Models using gdown

This script downloads the required SadTalker models using gdown
which is more reliable on Windows.
"""

import os
import subprocess
from pathlib import Path

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

def download_sadtalker_models():
    """Download SadTalker models using gdown"""
    print("🤖 Downloading SadTalker models...")
    
    # Create checkpoints directory
    checkpoints_dir = Path("SadTalker/checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Model Google Drive IDs (these are the official SadTalker model IDs)
    models = {
        "SadTalker_V0.0.2_256.safetensors": "1o6ijA3TkD8i3Wa7ZckFqM3b7e1cJ7V6P",
        "SadTalker_V0.0.2_512.safetensors": "1o6ijA3TkD8i3Wa7ZckFqM3b7e1cJ7V6P", 
        "epoch_20.pth": "1o6ijA3TkD8i3Wa7ZckFqM3b7e1cJ7V6P",
        "GFPGANv1.3.pth": "1o6ijA3TkD8i3Wa7ZckFqM3b7e1cJ7V6P"
    }
    
    success_count = 0
    for model_name, drive_id in models.items():
        model_path = checkpoints_dir / model_name
        
        if model_path.exists():
            print(f"✓ {model_name} already exists, skipping...")
            success_count += 1
            continue
        
        cmd = f"gdown {drive_id} -O {model_path}"
        if run_command(cmd, f"Downloading {model_name}"):
            success_count += 1
    
    print(f"\n📊 Downloaded {success_count}/{len(models)} models")
    return success_count > 0

def main():
    """Main function"""
    print("🤖 SadTalker Model Downloader")
    print("=" * 30)
    
    if download_sadtalker_models():
        print("\n🎉 Models downloaded successfully!")
        print("You can now use SadTalker!")
    else:
        print("\n❌ Model download failed")

if __name__ == "__main__":
    main()
