#!/usr/bin/env python3
"""
D-ID API Video Generation Example

This script demonstrates how to generate a talking video using the D-ID API.
"""

import requests
import time
import base64

API_KEY = "dGVhbTQwNGhhY2tjbXVAZ21haWwuY29t:VFAB5GD9hWE4FSYCQpBUA"  # Updated D-ID API key
BASE_URL = "https://api.d-id.com/talks"

# D-ID expects Basic Auth with base64 encoding
def get_headers():
    api_key_bytes = API_KEY.encode('utf-8')
    base64_key = base64.b64encode(api_key_bytes).decode('utf-8')
    return {
        "Authorization": f"Basic {base64_key}",
        "Content-Type": "application/json"
    }

def create_talk(image_url, text):
    # Check for direct public image URL
    if not (image_url.startswith("http://") or image_url.startswith("https://")):
        print("Error: Please provide a direct public image URL (e.g., Imgur, Dropbox direct link). Google Drive share links will NOT work.")
        return None
    data = {
        "source_url": image_url,
        "script": {
            "type": "text",
            "input": text,
            "provider": {
                "type": "microsoft",
                "voice_id": "en-US-JennyNeural"  # Example voice
            }
        }
    }
    response = requests.post(BASE_URL, headers=get_headers(), json=data)
    print(f"Status code: {response.status_code}")
    print("Response:")
    print(response.text)
    if response.status_code == 201:
        talk_id = response.json().get("id")
        return talk_id
    return None

def get_talk_status(talk_id):
    url = f"{BASE_URL}/{talk_id}"
    response = requests.get(url, headers=get_headers())
    print(f"Status code: {response.status_code}")
    print("Response:")
    print(response.text)
    if response.status_code == 200:
        return response.json()
    return None

def main():
    image_url = "https://drive.google.com/drive/u/2/home.jpg"
    text = "Hello, nice to meet you!"  # Must be at least 3 characters
    print(f"Using image URL: {image_url}")
    print(f"Using text: {text}")
    create_talk(image_url, text)

if __name__ == "__main__":
    main()
