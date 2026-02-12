#!/usr/bin/env python3
"""
Ultra-Simple Talking Video Generator

This version creates a basic talking video without any complex dependencies.
Just needs: gtts, opencv-python, numpy
"""

import os
import cv2
import numpy as np
from pathlib import Path

def create_talking_video(image_path, text, output_name="output"):
    """Create a simple talking video"""
    
    print(f"🎬 Creating talking video...")
    print(f"📸 Image: {image_path}")
    print(f"📝 Text: {text}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return False
    
    height, width = img.shape[:2]
    print(f"✓ Image loaded: {width}x{height}")
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate speech
    try:
        from gtts import gTTS
        print("🎵 Generating speech...")
        tts = gTTS(text=text, lang='en', slow=False)
        audio_path = output_dir / f"{output_name}.wav"
        tts.save(str(audio_path))
        print(f"✓ Speech saved: {audio_path}")
    except Exception as e:
        print(f"❌ Speech generation failed: {e}")
        return False
    
    # Create video
    print("🎥 Creating video...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 25
    video_path = output_dir / f"{output_name}.mp4"
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    
    # Calculate duration (rough estimate)
    word_count = len(text.split())
    duration = max(3, word_count * 0.5)  # Minimum 3 seconds
    frame_count = int(duration * fps)
    
    print(f"📊 Creating {frame_count} frames ({duration:.1f} seconds)")
    
    # Create frames with simple animation
    for i in range(frame_count):
        frame = img.copy()
        
        # Add progress indicator
        progress = i / frame_count
        bar_width = int(width * progress)
        cv2.rectangle(frame, (0, height-10), (bar_width, height), (0, 255, 0), -1)
        
        # Add text overlay
        text_overlay = f"Speaking: {text[:30]}{'...' if len(text) > 30 else ''}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 2
        
        # Position text
        (text_width, text_height), _ = cv2.getTextSize(text_overlay, font, font_scale, thickness)
        text_x = (width - text_width) // 2
        text_y = 30
        
        # Draw text background
        cv2.rectangle(frame, (text_x - 5, text_y - text_height - 5), 
                     (text_x + text_width + 5, text_y + 5), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, text_overlay, (text_x, text_y), font, font_scale, color, thickness)
        
        # Add simple mouth animation
        center_x = width // 2
        center_y = int(height * 0.7)
        mouth_size = int(15 + 10 * np.sin(progress * 20))
        cv2.ellipse(frame, (center_x, center_y), (mouth_size, 8), 0, 0, 180, (0, 0, 0), 2)
        
        out.write(frame)
    
    out.release()
    
    print(f"✅ Video created successfully!")
    print(f"🎥 Video: {video_path}")
    print(f"🎵 Audio: {audio_path}")
    
    return True

def main():
    """Main function"""
    print("🎬 Ultra-Simple Talking Video Generator")
    print("=" * 40)
    
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
        output_name = "talking_video"
    
    # Create video
    success = create_talking_video(image_path, text, output_name)
    
    if success:
        print("\n🎉 Success! Check the 'output' folder for your video!")
    else:
        print("\n❌ Failed to create video")

if __name__ == "__main__":
    main()
