#!/usr/bin/env python3
"""
Simplified Talking Video Generator using SadTalker + Google TTS

This version uses Google TTS (gTTS) which is more reliable and has fewer
dependency issues than Coqui TTS.

Installation:
pip install gtts torch torchvision torchaudio opencv-python pillow numpy gdown

Usage:
python talking_video_simple.py
"""

import os
import sys
import subprocess
import shutil
import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import cv2
import numpy as np
from PIL import Image

# TTS library
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
    print("✓ Google TTS (gTTS) available")
except ImportError:
    TTS_AVAILABLE = False
    print("❌ gTTS not available. Install with: pip install gtts")

class SimpleTalkingVideoGenerator:
    """
    Generate realistic talking videos using SadTalker + Google TTS
    """
    
    def __init__(self, 
                 sadtalker_path: str = "SadTalker",
                 output_dir: str = "output"):
        """
        Initialize the talking video generator
        
        Args:
            sadtalker_path: Path to SadTalker directory
            output_dir: Output directory for generated videos
        """
        self.sadtalker_path = Path(sadtalker_path)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        print("✓ Simple Talking Video Generator initialized!")
    
    def check_sadtalker_setup(self) -> bool:
        """
        Check if SadTalker is properly set up
        
        Returns:
            True if SadTalker is ready, False otherwise
        """
        if not self.sadtalker_path.exists():
            print(f"❌ SadTalker directory not found: {self.sadtalker_path}")
            print("Run: python setup_talking_video.py")
            return False
        
        # Check for required files
        required_files = [
            "inference.py",
            "src",
            "checkpoints"
        ]
        
        for file in required_files:
            if not (self.sadtalker_path / file).exists():
                print(f"❌ Required SadTalker file/directory not found: {file}")
                return False
        
        print("✓ SadTalker setup looks good!")
        return True
    
    def validate_image(self, image_path: str) -> bool:
        """
        Validate input image for SadTalker
        
        Args:
            image_path: Path to input image
            
        Returns:
            True if image is valid, False otherwise
        """
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return False
        
        try:
            # Load and validate image
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Invalid image file: {image_path}")
                return False
            
            height, width = img.shape[:2]
            
            # Check image dimensions
            if width < 256 or height < 256:
                print(f"❌ Image too small: {width}x{height}. Minimum: 256x256")
                return False
            
            # Check if image has a face (basic check)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                print("⚠️  Warning: No face detected in image. Results may be poor.")
            else:
                print(f"✓ Face detected in image. Found {len(faces)} face(s).")
            
            return True
            
        except Exception as e:
            print(f"❌ Error validating image: {e}")
            return False
    
    def generate_speech(self, text: str, output_audio: str) -> bool:
        """
        Generate speech from text using Google TTS
        
        Args:
            text: Text to convert to speech
            output_audio: Output audio file path
            
        Returns:
            True if successful, False otherwise
        """
        if not TTS_AVAILABLE:
            print("❌ Google TTS not available. Cannot generate speech.")
            return False
        
        try:
            print(f"Generating speech: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            # Generate speech using Google TTS
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(output_audio)
            
            # Verify output
            if os.path.exists(output_audio) and os.path.getsize(output_audio) > 0:
                print(f"✓ Speech generated successfully: {output_audio}")
                return True
            else:
                print(f"❌ Speech generation failed: {output_audio}")
                return False
                
        except Exception as e:
            print(f"❌ Error generating speech: {e}")
            return False
    
    def generate_talking_video(self, 
                             input_image: str, 
                             input_audio: str, 
                             output_video: str,
                             still_mode: bool = True,
                             preprocess: str = "full") -> bool:
        """
        Generate talking video using SadTalker
        
        Args:
            input_image: Path to input image
            input_audio: Path to input audio
            output_video: Path to output video
            still_mode: Use still mode for better results
            preprocess: Preprocessing mode
            
        Returns:
            True if successful, False otherwise
        """
        if not self.check_sadtalker_setup():
            return False
        
        try:
            print("Generating talking video with SadTalker...")
            
            # Prepare SadTalker command
            cmd = [
                "python", str(self.sadtalker_path / "inference.py"),
                "--driven_audio", input_audio,
                "--source_image", input_image,
                "--result_dir", str(self.output_dir / "temp_results"),
                "--preprocess", preprocess
            ]
            
            if still_mode:
                cmd.append("--still")
            
            # Run SadTalker
            print("Running SadTalker inference...")
            result = subprocess.run(cmd, 
                                  cwd=self.sadtalker_path,
                                  capture_output=True, 
                                  text=True, 
                                  check=False)
            
            if result.returncode != 0:
                print(f"❌ SadTalker failed with return code: {result.returncode}")
                print(f"Error output: {result.stderr}")
                return False
            
            # Find and move the generated video
            temp_results_dir = self.output_dir / "temp_results"
            if temp_results_dir.exists():
                for file in temp_results_dir.iterdir():
                    if file.suffix == ".mp4":
                        shutil.move(str(file), output_video)
                        print(f"✓ Video generated successfully: {output_video}")
                        
                        # Clean up temp directory
                        shutil.rmtree(temp_results_dir, ignore_errors=True)
                        return True
            
            print("❌ No video file found in SadTalker output")
            return False
            
        except Exception as e:
            print(f"❌ Error generating talking video: {e}")
            return False
    
    def process_single(self, 
                      input_image: str, 
                      text: str, 
                      output_name: str = None) -> Optional[str]:
        """
        Process a single image and text to generate talking video
        
        Args:
            input_image: Path to input image
            text: Text to speak
            output_name: Custom output name (optional)
            
        Returns:
            Path to generated video if successful, None otherwise
        """
        # Validate inputs
        if not self.validate_image(input_image):
            return None
        
        if not text.strip():
            print("❌ Text cannot be empty")
            return None
        
        # Generate output names
        if output_name is None:
            base_name = Path(input_image).stem
            output_name = f"{base_name}_talking"
        
        output_audio = str(self.output_dir / f"{output_name}.wav")
        output_video = str(self.output_dir / f"{output_name}.mp4")
        
        print(f"\n🎬 Processing: {Path(input_image).name}")
        print(f"📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"🎵 Audio: {output_audio}")
        print(f"🎥 Video: {output_video}")
        
        # Step 1: Generate speech
        if not self.generate_speech(text, output_audio):
            return None
        
        # Step 2: Generate talking video
        if not self.generate_talking_video(input_image, output_audio, output_video):
            return None
        
        return output_video
    
    def process_batch(self, 
                     image_text_pairs: List[Dict[str, str]], 
                     output_prefix: str = "batch") -> List[str]:
        """
        Process multiple image-text pairs
        
        Args:
            image_text_pairs: List of dicts with 'image' and 'text' keys
            output_prefix: Prefix for output files
            
        Returns:
            List of generated video paths
        """
        results = []
        
        print(f"\n🔄 Processing {len(image_text_pairs)} items in batch...")
        
        for i, pair in enumerate(image_text_pairs):
            print(f"\n--- Item {i+1}/{len(image_text_pairs)} ---")
            
            image_path = pair.get('image')
            text = pair.get('text')
            
            if not image_path or not text:
                print(f"❌ Skipping item {i+1}: missing image or text")
                continue
            
            output_name = f"{output_prefix}_{i+1:03d}"
            result = self.process_single(image_path, text, output_name)
            
            if result:
                results.append(result)
                print(f"✅ Item {i+1} completed: {result}")
            else:
                print(f"❌ Item {i+1} failed")
        
        print(f"\n🎉 Batch processing complete! {len(results)}/{len(image_text_pairs)} successful")
        return results

def setup_sadtalker():
    """Setup SadTalker (run this once)"""
    print("🔧 Setting up SadTalker...")
    
    # Clone SadTalker repository
    if not os.path.exists("SadTalker"):
        print("Cloning SadTalker repository...")
        subprocess.run([
            "git", "clone", "https://github.com/OpenTalker/SadTalker.git"
        ], check=True)
    else:
        print("✓ SadTalker directory already exists")
    
    # Install requirements
    print("Installing SadTalker requirements...")
    subprocess.run([
        "pip", "install", "-r", "requirements.txt"
    ], cwd="SadTalker", check=True)
    
    # Download models
    print("Downloading SadTalker models...")
    subprocess.run([
        "bash", "scripts/download_models.sh"
    ], cwd="SadTalker", check=True)
    
    print("✅ SadTalker setup complete!")

def main():
    """Main function with example usage"""
    parser = argparse.ArgumentParser(description="Generate realistic talking videos (Simple Version)")
    parser.add_argument("--image", "-i", help="Input image path")
    parser.add_argument("--text", "-t", help="Text to speak")
    parser.add_argument("--output", "-o", help="Output video name")
    parser.add_argument("--batch", "-b", help="Batch processing JSON file")
    parser.add_argument("--setup", action="store_true", help="Setup SadTalker")
    
    args = parser.parse_args()
    
    # Setup SadTalker if requested
    if args.setup:
        setup_sadtalker()
        return
    
    # Initialize generator
    generator = SimpleTalkingVideoGenerator()
    
    # Batch processing
    if args.batch:
        try:
            with open(args.batch, 'r') as f:
                batch_data = json.load(f)
            results = generator.process_batch(batch_data)
            print(f"\n🎉 Generated {len(results)} videos!")
        except Exception as e:
            print(f"❌ Error processing batch file: {e}")
        return
    
    # Single processing
    if args.image and args.text:
        result = generator.process_single(args.image, args.text, args.output)
        if result:
            print(f"\n🎉 Video generated successfully: {result}")
        else:
            print("\n❌ Video generation failed")
        return
    
    # Interactive mode
    print("🎬 Simple Talking Video Generator - Interactive Mode")
    print("=" * 50)
    
    # Get input image
    while True:
        image_path = input("\n📸 Enter image path (or 'quit' to exit): ").strip()
        if image_path.lower() == 'quit':
            return
        if generator.validate_image(image_path):
            break
        print("Please enter a valid image path.")
    
    # Get text
    text = input("📝 Enter text to speak: ").strip()
    if not text:
        print("❌ Text cannot be empty")
        return
    
    # Get output name
    output_name = input("📁 Enter output name (optional): ").strip()
    if not output_name:
        output_name = None
    
    # Process
    result = generator.process_single(image_path, text, output_name)
    
    if result:
        print(f"\n🎉 Video generated successfully: {result}")
    else:
        print("\n❌ Video generation failed")

if __name__ == "__main__":
    main()
