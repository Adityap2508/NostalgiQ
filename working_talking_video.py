#!/usr/bin/env python3
"""
Working Talking Video Generator - No SadTalker Required

This version uses a simpler approach that works immediately without
requiring SadTalker setup or model downloads.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import json
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# TTS library
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
    print("✓ Google TTS (gTTS) available")
except ImportError:
    TTS_AVAILABLE = False
    print("❌ gTTS not available. Install with: pip install gtts")

class WorkingTalkingVideoGenerator:
    """
    Generate talking videos using a simpler approach
    """
    
    def __init__(self, output_dir: str = "output"):
        """Initialize the generator"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        print("✓ Working Talking Video Generator initialized!")
    
    def validate_image(self, image_path: str) -> bool:
        """Validate input image"""
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return False
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Invalid image file: {image_path}")
                return False
            
            height, width = img.shape[:2]
            if width < 256 or height < 256:
                print(f"❌ Image too small: {width}x{height}. Minimum: 256x256")
                return False
            
            print(f"✓ Image validated: {width}x{height}")
            return True
            
        except Exception as e:
            print(f"❌ Error validating image: {e}")
            return False
    
    def generate_speech(self, text: str, output_audio: str) -> bool:
        """Generate speech using Google TTS"""
        if not TTS_AVAILABLE:
            print("❌ Google TTS not available")
            return False
        
        try:
            print(f"Generating speech: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(output_audio)
            
            if os.path.exists(output_audio) and os.path.getsize(output_audio) > 0:
                print(f"✓ Speech generated: {output_audio}")
                return True
            else:
                print(f"❌ Speech generation failed")
                return False
                
        except Exception as e:
            print(f"❌ Error generating speech: {e}")
            return False
    
    def create_simple_talking_video(self, image_path: str, audio_path: str, output_video: str) -> bool:
        """Create a simple talking video without SadTalker"""
        try:
            print("Creating simple talking video...")
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                print("❌ Could not load image")
                return False
            
            # Get image dimensions
            height, width = img.shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 25
            out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            
            # Calculate duration (rough estimate: 150 words per minute)
            word_count = len(audio_path.split())  # Rough estimate
            duration = max(2, word_count * 0.4)  # Minimum 2 seconds
            frame_count = int(duration * fps)
            
            print(f"Creating {frame_count} frames ({duration:.1f} seconds)")
            
            # Create frames with subtle animation
            for i in range(frame_count):
                # Create a copy of the image
                frame = img.copy()
                
                # Add subtle mouth movement simulation
                progress = i / frame_count
                
                # Simple mouth animation (draw a moving ellipse)
                center_x = width // 2
                center_y = int(height * 0.7)
                
                # Mouth size varies with "speech"
                mouth_size = int(20 + 10 * np.sin(progress * 20))
                mouth_height = int(8 + 4 * np.sin(progress * 15))
                
                # Draw mouth
                cv2.ellipse(frame, (center_x, center_y), (mouth_size, mouth_height), 
                           0, 0, 180, (0, 0, 0), 2)
                
                # Add text overlay
                text = "Speaking..."
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                color = (255, 255, 255)
                thickness = 2
                
                # Get text size
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Position text at bottom
                text_x = (width - text_width) // 2
                text_y = height - 20
                
                # Draw text background
                cv2.rectangle(frame, (text_x - 10, text_y - text_height - 10), 
                             (text_x + text_width + 10, text_y + 10), (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
                
                # Write frame
                out.write(frame)
            
            # Release video writer
            out.release()
            
            print(f"✓ Simple talking video created: {output_video}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating video: {e}")
            return False
    
    def process_single(self, input_image: str, text: str, output_name: str = None) -> Optional[str]:
        """Process a single image and text"""
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
        
        # Step 2: Create simple talking video
        if not self.create_simple_talking_video(input_image, output_audio, output_video):
            return None
        
        return output_video

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Working Talking Video Generator")
    parser.add_argument("--image", "-i", help="Input image path")
    parser.add_argument("--text", "-t", help="Text to speak")
    parser.add_argument("--output", "-o", help="Output video name")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = WorkingTalkingVideoGenerator()
    
    # Single processing
    if args.image and args.text:
        result = generator.process_single(args.image, args.text, args.output)
        if result:
            print(f"\n🎉 Video generated successfully: {result}")
        else:
            print("\n❌ Video generation failed")
        return
    
    # Interactive mode
    print("🎬 Working Talking Video Generator - Interactive Mode")
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
