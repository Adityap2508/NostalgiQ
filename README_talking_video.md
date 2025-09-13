# Talking Video Generator

## Quick Start

1. **Setup (run once):**
   ```bash
   python setup_talking_video.py
   ```

2. **Generate a talking video:**
   ```bash
   python talking_video_generator.py --image person.jpg --text "Hello world!"
   ```

3. **Batch processing:**
   ```bash
   python talking_video_generator.py --batch example_batch.json
   ```

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- At least 8GB RAM
- 10GB free disk space

## Input Image Requirements

- Frontal face photo
- Good lighting
- High resolution (256x256 minimum)
- Common formats: JPG, PNG

## Output

- Generated videos saved in `output/` directory
- MP4 format
- Audio synchronized with lip movements

## Troubleshooting

- If TTS fails: Try different TTS models with `--list-models`
- If SadTalker fails: Check GPU memory and try `--still` mode
- For poor quality: Use higher resolution input images
