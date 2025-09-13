#!/usr/bin/env python3
"""
Enhanced Simple Talking Video Generator

This creates a more realistic talking video with better mouth animation
without requiring SadTalker's large model downloads.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from gtts import gTTS
import math

class EnhancedTalkingVideo:
    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def create_realistic_mouth_animation(self, frame, width, height, frame_num, total_frames, text_length):
        """Create more realistic mouth animation based on speech patterns"""
        # Calculate animation progress
        progress = frame_num / total_frames
        
        # Create speech-like rhythm (more realistic than simple sine wave)
        speech_rhythm = 0
        for i in range(3):  # Multiple frequency components
            freq = (i + 1) * 8 + text_length * 0.5
            speech_rhythm += math.sin(progress * freq * math.pi) / (i + 1)
        
        # Add pauses (mouth closed) at certain intervals
        pause_factor = 1.0
        if int(progress * 20) % 4 == 0:  # Pause every 4th beat
            pause_factor = 0.3
        
        # Calculate mouth size and position
        base_mouth_size = 20
        mouth_variation = 15 * speech_rhythm * pause_factor
        mouth_width = max(5, int(base_mouth_size + mouth_variation))
        mouth_height = max(3, int(8 + mouth_variation * 0.3))
        
        # Mouth position (slightly below center)
        mouth_x = width // 2
        mouth_y = int(height * 0.65)
        
        # Draw mouth with gradient effect
        for i in range(3):
            color_intensity = 255 - i * 30
            thickness = 3 - i
            if thickness > 0:
                cv2.ellipse(frame, (mouth_x, mouth_y), 
                           (mouth_width - i*2, mouth_height - i), 
                           0, 0, 180, (0, 0, color_intensity), thickness)
        
        # Add subtle lip movement
        lip_offset = int(mouth_variation * 0.1)
        cv2.ellipse(frame, (mouth_x + lip_offset, mouth_y - 2), 
                   (mouth_width - 2, 2), 0, 0, 180, (0, 0, 200), 1)
        
        return frame
    
    def add_eye_blink(self, frame, width, height, frame_num):
        """Add occasional eye blinking"""
        # Blink every 60-90 frames
        if frame_num % 75 == 0:
            eye_y = int(height * 0.4)
            eye_x1 = int(width * 0.4)
            eye_x2 = int(width * 0.6)
            
            # Draw closed eyes
            cv2.line(frame, (eye_x1 - 10, eye_y), (eye_x1 + 10, eye_y), (0, 0, 0), 3)
            cv2.line(frame, (eye_x2 - 10, eye_y), (eye_x2 + 10, eye_y), (0, 0, 0), 3)
        
        return frame
    
    def add_subtle_head_movement(self, frame, frame_num):
        """Add subtle head movement for realism"""
        # Very subtle rotation and translation
        angle = math.sin(frame_num * 0.02) * 0.5  # Max 0.5 degree rotation
        tx = math.sin(frame_num * 0.03) * 2  # Max 2 pixel translation
        ty = math.cos(frame_num * 0.025) * 1  # Max 1 pixel translation
        
        # Apply transformation
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        M[0, 2] += tx
        M[1, 2] += ty
        
        frame = cv2.warpAffine(frame, M, (w, h))
        return frame
    
    def create_video(self, image_path, text, output_name):
        """Create enhanced talking video"""
        print("🎬 Creating enhanced talking video...")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        height, width = img.shape[:2]
        print(f"✓ Image loaded: {width}x{height}")
        
        # Calculate video duration based on text length
        text_length = len(text.split())
        duration = max(2.0, text_length * 0.4)  # 0.4 seconds per word, minimum 2 seconds
        fps = 25
        total_frames = int(duration * fps)
        
        print(f"📊 Creating {total_frames} frames ({duration:.1f} seconds)")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = self.output_dir / f"{output_name}.mp4"
        out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
        
        # Generate speech
        print("🎵 Generating speech...")
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            output_audio = self.output_dir / f"{output_name}.wav"
            tts.save(str(output_audio))
            print(f"✓ Speech saved: {output_audio}")
        except Exception as e:
            print(f"❌ Speech generation failed: {e}")
            return None
        
        # Create frames with enhanced animation
        for i in range(total_frames):
            frame = img.copy()
            
            # Add mouth animation
            frame = self.create_realistic_mouth_animation(frame, width, height, i, total_frames, text_length)
            
            # Add eye blinking
            frame = self.add_eye_blink(frame, width, height, i)
            
            # Add subtle head movement
            frame = self.add_subtle_head_movement(frame, i)
            
            # Write frame
            out.write(frame)
        
        # Release video writer
        out.release()
        
        print(f"✅ Enhanced video created: {output_video}")
        return str(output_video)

def main():
    """Main function"""
    print("🎬 Enhanced Simple Talking Video Generator")
    print("=" * 45)
    
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
        output_name = "enhanced_talking_video"
    
    # Create video
    generator = EnhancedTalkingVideo()
    result = generator.create_video(image_path, text, output_name)
    
    if result:
        print(f"\n🎉 Enhanced talking video created: {result}")
        print("🎵 Audio file also created in the same directory")
    else:
        print("\n❌ Video creation failed")

if __name__ == "__main__":
    main()
