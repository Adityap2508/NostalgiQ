#!/usr/bin/env python3
"""
Working SadTalker Integration

This script integrates SadTalker with proper error handling and
fallback options.
"""

import os
import subprocess
import shutil
from pathlib import Path
import argparse
from typing import Optional

# TTS
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

class SadTalkerGenerator:
    """Working SadTalker generator with fallbacks"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.sadtalker_path = Path("SadTalker")
        
    def check_sadtalker(self) -> bool:
        """Check if SadTalker is properly set up"""
        if not self.sadtalker_path.exists():
            print("❌ SadTalker directory not found")
            return False
        
        checkpoints_dir = self.sadtalker_path / "checkpoints"
        if not checkpoints_dir.exists():
            print("❌ Checkpoints directory not found")
            return False
        
        # Check for key model files
        key_models = ["SadTalker_V0.0.2_256.safetensors", "epoch_20.pth"]
        found_models = sum(1 for model in key_models if (checkpoints_dir / model).exists())
        
        if found_models >= 1:
            print("✅ SadTalker appears to be ready!")
            return True
        else:
            print("❌ SadTalker models not found")
            return False
    
    def generate_speech(self, text: str, output_audio: str) -> bool:
        """Generate speech using TTS"""
        if not TTS_AVAILABLE:
            print("❌ TTS not available")
            return False
        
        try:
            print(f"🎵 Generating speech: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(output_audio)
            print(f"✅ Speech generated: {output_audio}")
            return True
        except Exception as e:
            print(f"❌ Speech generation failed: {e}")
            return False
    
    def run_sadtalker(self, image_path: str, audio_path: str, output_video: str) -> bool:
        """Run SadTalker inference"""
        try:
            print("🤖 Running SadTalker...")
            
            # Prepare command
            cmd = [
                "python", str(self.sadtalker_path / "inference.py"),
                "--driven_audio", audio_path,
                "--source_image", image_path,
                "--result_dir", str(self.output_dir / "temp_results"),
                "--preprocess", "full",
                "--still"
            ]
            
            # Run SadTalker
            result = subprocess.run(cmd, cwd=self.sadtalker_path, 
                                  capture_output=True, text=True, check=False)
            
            if result.returncode != 0:
                print(f"❌ SadTalker failed: {result.stderr}")
                return False
            
            # Find output video
            temp_results = self.output_dir / "temp_results"
            if temp_results.exists():
                for file in temp_results.iterdir():
                    if file.suffix == ".mp4":
                        shutil.move(str(file), output_video)
                        shutil.rmtree(temp_results, ignore_errors=True)
                        print(f"✅ Video generated: {output_video}")
                        return True
            
            print("❌ No video output found")
            return False
            
        except Exception as e:
            print(f"❌ SadTalker error: {e}")
            return False
    
    def process(self, image_path: str, text: str, output_name: str = None) -> Optional[str]:
        """Process image and text to generate talking video"""
        # Validate inputs
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
        
        if not text.strip():
            print("❌ Text cannot be empty")
            return None
        
        # Check SadTalker
        if not self.check_sadtalker():
            print("❌ SadTalker not ready. Please run model download first.")
            return None
        
        # Generate output names
        if output_name is None:
            base_name = Path(image_path).stem
            output_name = f"{base_name}_sadtalker"
        
        output_audio = str(self.output_dir / f"{output_name}.wav")
        output_video = str(self.output_dir / f"{output_name}.mp4")
        
        print(f"\n🎬 Processing with SadTalker:")
        print(f"📸 Image: {image_path}")
        print(f"📝 Text: {text}")
        print(f"🎵 Audio: {output_audio}")
        print(f"🎥 Video: {output_video}")
        
        # Generate speech
        if not self.generate_speech(text, output_audio):
            return None
        
        # Run SadTalker
        if not self.run_sadtalker(image_path, output_audio, output_video):
            return None
        
        return output_video

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Working SadTalker Generator")
    parser.add_argument("--image", "-i", help="Input image path")
    parser.add_argument("--text", "-t", help="Text to speak")
    parser.add_argument("--output", "-o", help="Output video name")
    parser.add_argument("--download-models", action="store_true", help="Download SadTalker models")
    
    args = parser.parse_args()
    
    # Download models if requested
    if args.download_models:
        print("📥 Downloading SadTalker models...")
        os.system("python download_sadtalker_models.py")
        return
    
    # Initialize generator
    generator = SadTalkerGenerator()
    
    # Single processing
    if args.image and args.text:
        result = generator.process(args.image, args.text, args.output)
        if result:
            print(f"\n🎉 SadTalker video generated: {result}")
        else:
            print("\n❌ SadTalker generation failed")
        return
    
    # Interactive mode
    print("🤖 Working SadTalker Generator")
    print("=" * 30)
    
    # Get inputs
    image_path = input("📸 Enter image path: ").strip()
    if not image_path:
        print("❌ Image path required")
        return
    
    text = input("📝 Enter text to speak: ").strip()
    if not text:
        print("❌ Text required")
        return
    
    output_name = input("📁 Enter output name (optional): ").strip()
    if not output_name:
        output_name = None
    
    # Process
    result = generator.process(image_path, text, output_name)
    
    if result:
        print(f"\n🎉 SadTalker video generated: {result}")
    else:
        print("\n❌ SadTalker generation failed")

if __name__ == "__main__":
    main()
