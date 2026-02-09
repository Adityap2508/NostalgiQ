#!/usr/bin/env python3
"""
HeyGen AI Video Generation Script

This script uses the HeyGen API to generate talking videos using your API key.
"""

import requests
import json
from pathlib import Path

API_KEY = "NDUzOGE5MDcxMmU1NDZhNDhiMDNmN2MxMzU4Y2ZjZDMtMTc1Nzc0NDIwNA=="
BASE_URL = "https://api.heygen.com/v2/video/generate"

class HeyGenVideoGenerator:
    def __init__(self, api_key=API_KEY, output_dir="output"):
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_video(self, image_url, text, output_name=None):
        print(f"🎬 Generating video with HeyGen AI...")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "video_inputs": [
                {
                    "avatar_id": "dcb591bfd0084443b423b3faa5b8c8f3",
                    "avatar_style": "photo",
                    "image_url": image_url,
                    "script": {
                        "type": "text",
                        "input": text,
                        "voice_id": "2f89b0ed799d441c894d832c966beca2"
                    }
                }
            ],
            "config": {
                "output_format": "mp4"
            }
        }
        response = requests.post(BASE_URL, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            result = response.json()
            video_url = result.get("video_url")
            if video_url:
                print(f"✓ Video generated. Downloading from: {video_url}")
                return self._download_video(video_url, output_name or "heygen_video.mp4")
            else:
                print("❌ No video URL returned.")
        else:
            print(f"❌ API error: {response.status_code} {response.text}")
        return None

    # _upload_to_imgur removed; always use public image URL

    # _encode_image removed; HeyGen expects image_url, not base64

    def _download_video(self, video_url, output_name):
        output_path = self.output_dir / output_name
        r = requests.get(video_url, stream=True)
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"✅ Video saved: {output_path}")
        return str(output_path)

def main():
    print("🖼️ HeyGen AI Video Generator")
    print("=" * 40)
    print("Paste a public image URL (e.g., from Imgur, Dropbox, etc.)")
    image_url = input("Enter public image URL: ").strip()
    text = input("Enter text to speak: ").strip()
    output_name = input("Enter output video name (optional): ").strip() or None
    generator = HeyGenVideoGenerator()
    generator.generate_video(image_url, text, output_name)

if __name__ == "__main__":
    main()
