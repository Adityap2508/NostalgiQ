#!/usr/bin/env python3
"""
Realistic Talking Video Generator using SadTalker + Coqui TTS

This script generates realistic talking head videos from a static image and text input.
It combines Coqui TTS for speech synthesis and SadTalker for video generation.

Installation:
pip install gdown torch torchvision torchaudio TTS opencv-python pillow numpy

Usage:
python talking_video_generator.py

Features:
- Text-to-Speech using Coqui TTS
- Realistic talking head animation using SadTalker
- Multiple TTS model options
- Batch processing support
- Progress tracking and error handling
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

# TTS libraries - try multiple options
TTS_AVAILABLE = False
TTS_TYPE = None

# Try Coqui TTS first
try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
    TTS_TYPE = "coqui"
    print("✓ Coqui TTS available")
except ImportError:
    pass

# Try gTTS as fallback
if not TTS_AVAILABLE:
    try:
        from gtts import gTTS
        import pygame
        TTS_AVAILABLE = True
        TTS_TYPE = "gtts"
        print("✓ Google TTS (gTTS) available")
    except ImportError:
        pass

# Try pyttsx3 as another fallback
if not TTS_AVAILABLE:
    try:
        import pyttsx3
        TTS_AVAILABLE = True
        TTS_TYPE = "pyttsx3"
        print("✓ pyttsx3 TTS available")
    except ImportError:
        pass

if not TTS_AVAILABLE:
    print("Warning: No TTS library available. Install one of: pip install TTS, pip install gtts, pip install pyttsx3")

class TalkingVideoGenerator:
    """
    Generate realistic talking videos using SadTalker + Coqui TTS
    """
    
    def __init__(self, 
                 sadtalker_path: str = "SadTalker",
                 output_dir: str = "output",
                 tts_model: str = "tts_models/en/ljspeech/tacotron2-DDC"):
        """
        Initialize the talking video generator
        
        Args:
            sadtalker_path: Path to SadTalker directory
            output_dir: Output directory for generated videos
            tts_model: TTS model to use
        """
        self.sadtalker_path = Path(sadtalker_path)
        self.output_dir = Path(output_dir)
        self.tts_model = tts_model
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize TTS
        self.tts = None
        if TTS_AVAILABLE:
            self._initialize_tts()
        
        # Available TTS models
        self.available_tts_models = [
            "tts_models/en/ljspeech/tacotron2-DDC",
            "tts_models/en/ljspeech/fast_pitch",
            "tts_models/en/vctk/vits",
            "tts_models/en/sam/tacotron-DDC",
            "tts_models/multilingual/multi-dataset/your_tts"
        ]
        
        print("✓ Talking Video Generator initialized!")
    
    def _initialize_tts(self):
        """Initialize TTS model based on available library"""
        if TTS_TYPE == "coqui":
            try:
                print(f"Loading Coqui TTS model: {self.tts_model}")
                self.tts = TTS(model_name=self.tts_model, progress_bar=True, gpu=False)
                print("✓ Coqui TTS model loaded successfully!")
            except Exception as e:
                print(f"Error loading Coqui TTS model: {e}")
                print("Falling back to default model...")
                try:
                    self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", 
                                  progress_bar=True, gpu=False)
                    print("✓ Default Coqui TTS model loaded!")
                except Exception as e2:
                    print(f"Failed to load Coqui TTS model: {e2}")
                    self.tts = None
        
        elif TTS_TYPE == "gtts":
            print("✓ Google TTS (gTTS) initialized!")
            self.tts = "gtts"  # Marker for gTTS
            
        elif TTS_TYPE == "pyttsx3":
            try:
                import pyttsx3
                self.tts = pyttsx3.init()
                print("✓ pyttsx3 TTS initialized!")
            except Exception as e:
                print(f"Failed to initialize pyttsx3: {e}")
                self.tts = None
        
        else:
            print("❌ No TTS library available")
            self.tts = None
    
    def check_sadtalker_setup(self) -> bool:
        """
        Check if SadTalker is properly set up
        
        Returns:
            True if SadTalker is ready, False otherwise
        """
        if not self.sadtalker_path.exists():
            print(f"❌ SadTalker directory not found: {self.sadtalker_path}")
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
        Generate speech from text using available TTS library
        
        Args:
            text: Text to convert to speech
            output_audio: Output audio file path
            
        Returns:
            True if successful, False otherwise
        """
        if not TTS_AVAILABLE or self.tts is None:
            print("❌ TTS not available. Cannot generate speech.")
            return False
        
        try:
            print(f"Generating speech: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            if TTS_TYPE == "coqui":
                # Coqui TTS
                self.tts.tts_to_file(text=text, file_path=output_audio)
                
            elif TTS_TYPE == "gtts":
                # Google TTS
                from gtts import gTTS
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(output_audio)
                
            elif TTS_TYPE == "pyttsx3":
                # pyttsx3 TTS
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)  # Speed of speech
                engine.setProperty('volume', 0.9)  # Volume level
                
                # Save to file
                engine.save_to_file(text, output_audio)
                engine.runAndWait()
            
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
        
        # Clean up audio file if requested
        # os.remove(output_audio)  # Uncomment to remove audio file
        
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
    
    def list_available_tts_models(self):
        """List available TTS models"""
        print("\nAvailable TTS Models:")
        for i, model in enumerate(self.available_tts_models, 1):
            marker = "✓" if model == self.tts_model else " "
            print(f"  {marker} {i}. {model}")
    
    def change_tts_model(self, model_name: str) -> bool:
        """
        Change TTS model
        
        Args:
            model_name: Name of TTS model to use
            
        Returns:
            True if successful, False otherwise
        """
        if not TTS_AVAILABLE:
            print("❌ TTS not available")
            return False
        
        try:
            print(f"Changing TTS model to: {model_name}")
            self.tts = TTS(model_name=model_name, progress_bar=True, gpu=False)
            self.tts_model = model_name
            print("✓ TTS model changed successfully!")
            return True
        except Exception as e:
            print(f"❌ Error changing TTS model: {e}")
            return False

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
    parser = argparse.ArgumentParser(description="Generate realistic talking videos")
    parser.add_argument("--image", "-i", help="Input image path")
    parser.add_argument("--text", "-t", help="Text to speak")
    parser.add_argument("--output", "-o", help="Output video name")
    parser.add_argument("--batch", "-b", help="Batch processing JSON file")
    parser.add_argument("--setup", action="store_true", help="Setup SadTalker")
    parser.add_argument("--list-models", action="store_true", help="List available TTS models")
    parser.add_argument("--tts-model", help="TTS model to use")
    
    args = parser.parse_args()
    
    # Setup SadTalker if requested
    if args.setup:
        setup_sadtalker()
        return
    
    # Initialize generator
    generator = TalkingVideoGenerator()
    
    # List TTS models if requested
    if args.list_models:
        generator.list_available_tts_models()
        return
    
    # Change TTS model if requested
    if args.tts_model:
        generator.change_tts_model(args.tts_model)
    
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
    print("🎬 Talking Video Generator - Interactive Mode")
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
