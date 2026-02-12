#!/usr/bin/env python3
"""
Wav2Lip Setup Script

This script automatically downloads and sets up Wav2Lip in the SadTalker/Wav2Lip directory, including the model checkpoint.
"""

import os
import subprocess
from pathlib import Path
import requests
from requests.auth import HTTPBasicAuth

WAV2LIP_REPO = "https://github.com/Rudrabha/Wav2Lip.git"


def clone_wav2lip(dest_dir):
    print(f"📦 Cloning Wav2Lip repo into {dest_dir} ...")
    if not dest_dir.exists():
        subprocess.run(["git", "clone", WAV2LIP_REPO, str(dest_dir)], check=True)
        print("✓ Wav2Lip repo cloned.")
    else:
        print("✓ Wav2Lip repo already exists, skipping clone.")


def download_checkpoint(checkpoint_path):
    print(f"📥 Downloading Wav2Lip model checkpoint to {checkpoint_path} ...")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        print("✓ Checkpoint already exists, skipping download.")
        return True
    print("\n⚠️ Wav2Lip repo cloned. Please follow the official Wav2Lip README to manually download the model checkpoint (wav2lip_gan.pth) and place it in:")
    print(f"  {checkpoint_path}")
    print("You can find download instructions in the repo: https://github.com/Rudrabha/Wav2Lip")
    if checkpoint_path.exists():
        print("✓ Checkpoint already exists.")
        return True
    else:
        print("❌ Checkpoint not found. Manual download required.")
        return False


def main():
    print("🤖 Wav2Lip Setup Script")
    print("=" * 30)
    sadtalker_dir = Path("SadTalker")
    wav2lip_dir = sadtalker_dir / "Wav2Lip"
    checkpoints_dir = wav2lip_dir / "checkpoints"
    checkpoint_path = checkpoints_dir / "wav2lip_gan.pth"

    clone_wav2lip(wav2lip_dir)
    success = download_checkpoint(checkpoint_path)

    if success:
        print("\n🎉 Wav2Lip setup complete!")
        print("You can now use Wav2Lip for lip-sync video generation.")
    else:
        print("\n❌ Wav2Lip setup failed.")

if __name__ == "__main__":
    main()
