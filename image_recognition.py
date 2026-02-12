#!/usr/bin/env python3
"""
Enhanced Image Recognition Script

This script performs detailed image recognition using OpenCV and DeepFace, with step-by-step analysis and reporting.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from gtts import gTTS
import subprocess

class EnhancedImageRecognition:
    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)


    def animate_mouth(self, frame, face_area, frame_num, total_frames, text_length):
        """Animate mouth region based on speech rhythm"""
        x, y, w, h = face_area
        # Mouth position: lower center of face
        mouth_x = x + w // 2
        mouth_y = y + int(h * 0.7)
        # Animation progress
        progress = frame_num / total_frames
        # Speech rhythm
        rhythm = 0
        for i in range(3):
            freq = (i + 1) * 8 + text_length * 0.5
            rhythm += np.sin(progress * freq * np.pi) / (i + 1)
        pause = 1.0
        if int(progress * 20) % 4 == 0:
            pause = 0.3
        base_size = w // 8
        mouth_variation = int(base_size * 0.7 * rhythm * pause)
        mouth_width = max(5, base_size + mouth_variation)
        mouth_height = max(3, int(base_size * 0.4 + mouth_variation * 0.3))
        # Draw mouth
        for i in range(2):
            color_intensity = 255 - i * 40
            thickness = 2 - i
            if thickness > 0:
                cv2.ellipse(frame, (mouth_x, mouth_y), (mouth_width - i*2, mouth_height - i), 0, 0, 180, (0, 0, color_intensity), thickness)
        # Lip line
        lip_offset = int(mouth_variation * 0.1)
        cv2.ellipse(frame, (mouth_x + lip_offset, mouth_y - 2), (mouth_width - 2, 2), 0, 0, 180, (0, 0, 200), 1)
        return frame

    def generate_audio(self, text, output_audio):
        print("🎵 Generating speech audio...")
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(output_audio)
        print(f"✓ Audio saved: {output_audio}")
        return output_audio

    def run_wav2lip(self, image_path, audio_path, output_path):
        print("🦷 Running Wav2Lip for lip-sync...")
        # Wav2Lip expects a video as input; create a video from the image
        temp_video = str(self.output_dir / "temp_input.mp4")
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        fps = 25
        duration = self.get_audio_duration(audio_path)
        total_frames = int(duration * fps)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
        for _ in range(total_frames):
            out.write(img)
        out.release()
        # Run Wav2Lip inference (assumes Wav2Lip repo is in SadTalker/Wav2Lip)
        wav2lip_script = os.path.join("SadTalker", "Wav2Lip", "inference.py")
        checkpoint_path = os.path.join("SadTalker", "Wav2Lip", "checkpoints", "wav2lip_gan.pth")
        command = [
            "python", wav2lip_script,
            "--checkpoint_path", checkpoint_path,
            "--face", temp_video,
            "--audio", audio_path,
            "--outfile", output_path
        ]
        print("🔧 Running:", " ".join(command))
        subprocess.run(command, check=True)
        print(f"✅ Wav2Lip output saved: {output_path}")
        return output_path

    def get_audio_duration(self, audio_path):
        import wave
        with wave.open(audio_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)

    def analyze_image_with_wav2lip(self, image_path, text, output_name=None):
        print(f"🔍 Creating lip-sync video with Wav2Lip: {image_path}")
        if not output_name:
            output_name = f"{Path(image_path).stem}_wav2lip"
        output_audio = str(self.output_dir / f"{output_name}.wav")
        output_video = str(self.output_dir / f"{output_name}.mp4")
        self.generate_audio(text, output_audio)
        self.run_wav2lip(image_path, output_audio, output_video)
        print(f"🎬 Lip-sync video created: {output_video}")
        return output_video

    def run_batch(self, input_dir):
        print(f"📂 Running batch image recognition in: {input_dir}")
        input_dir = Path(input_dir)
        images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
        print(f"Found {len(images)} images.")
        for img_path in images:
            self.analyze_image(str(img_path))

def main():
    print("🖼️ Image Recognition & Wav2Lip Lip-Sync Script")
    print("=" * 45)
    mode = input("Select mode: (1) Single image, (2) Batch folder, (3) Wav2Lip lip-sync video: ").strip()
    recognizer = EnhancedImageRecognition()
    if mode == "1":
        image_path = input("Enter image path: ").strip()
        recognizer.analyze_image(image_path)
    elif mode == "2":
        input_dir = input("Enter input folder path: ").strip()
        recognizer.run_batch(input_dir)
    elif mode == "3":
        image_path = input("Enter image path: ").strip()
        text = input("Enter text to lip-sync: ").strip()
        recognizer.analyze_image_with_wav2lip(image_path, text)
    else:
        print("❌ Invalid mode selected.")

if __name__ == "__main__":
    main()
