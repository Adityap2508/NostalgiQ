import requests

API_KEY = "NDUzOGE5MDcxMmU1NDZhNDhiMDNmN2MxMzU4Y2ZjZDMtMTc1Nzc0NDIwNA=="
BASE_URL = "https://api.heygen.com/v2/video/generate"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "video_inputs": [
        {
            "avatar_id": "dcb591bfd0084443b423b3faa5b8c8f3",
            "avatar_style": "photo",
            "image_url": "https://drive.google.com/uc?export=view&id=1GJJ1n3rnrP8hN2jFAAUoJc7dM1zMi2IE",
            "voice": {
                "type": "text",
                "voice_id": "2f89b0ed799d441c894d832c966beca2",
                "input_text": "Hello there!"
            },
            "script": {
                "type": "text",
                "input": "Hello there!"
            }
        }
    ],
    "config": {
        "output_format": "mp4",
        "resolution": "sd"  # Request standard definition for free plan
    }
}

response = requests.post(BASE_URL, headers=headers, json=data)
print(f"Status code: {response.status_code}")
print("Response:")
print(response.text)