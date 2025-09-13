
# CM: Comprehensive Media AI Toolkit

This repository provides a suite of Python tools for face analysis, talking video generation, personality prediction, and text/audio/image conversion. It integrates state-of-the-art models and APIs (SadTalker, HeyGen, D-ID, DeepFace, Gemini, Whisper) for research and creative projects.

---

## 📦 Modules & Features

### 1. Face Analysis Pipeline (`face_pipeline.py`)
- Detects faces in images/videos (InsightFace, DeepFace, MediaPipe)
- Clusters identities, estimates age, extracts facial landmarks
- Scene description (CLIP), object detection (YOLOv8), OCR (EasyOCR)
- Outputs cropped faces and `metadata.json` with full analysis

### 2. Talking Video Generation
- **SadTalker**: Realistic talking head videos from image + text/audio (`talking_video_generator.py`)
- **HeyGen API**: Cloud-based talking video from public image URL + text (`heygen_video.py`)
- **D-ID API**: Cloud-based talking video from public image URL + text/audio (`did_api_test.py`)

### 3. Personality Prediction (`personality_predictor.py`)
- Predicts personality traits from text using NLP models (transformers, scikit-learn)

### 4. Text/Audio/Image Conversion (`Text_conversion.py`)
- Extracts text and dates from .txt or image files (Gemini Vision)
- Converts speech to text (Whisper)

---

## 🚀 Installation

**Recommended:** Python 3.8+ and CUDA-compatible GPU for best performance.

Install core dependencies:
```bash
pip install -r requirements_talking_video.txt
pip install -r requirements_personality.txt
pip install insightface mediapipe deepface ultralytics easyocr clip-by-openai opencv-python scikit-learn torch torchvision pillow numpy
```

For SadTalker, see `SadTalker/README.md` and `WINDOWS_SETUP.md` for troubleshooting and manual model downloads.

---

## 🖥️ Usage

### Face Analysis
```bash
python face_pipeline.py --input input_media --output output
```

### Talking Video Generation
- **SadTalker:**
  ```bash
  python talking_video_generator.py --image person.jpg --text "Hello world!"
  ```
- **HeyGen API:**
  ```bash
  python heygen_video.py
  # Follow prompts for image URL and text
  ```
- **D-ID API:**
  ```bash
  python did_api_test.py
  # Edit script to set image_url and text
  ```

### Personality Prediction
```bash
python personality_predictor.py --text "I love meeting new people."
```

### Text/Audio/Image Conversion
```bash
python Text_conversion.py
# See script for usage details
```

---

## 🌐 API Integrations

### D-ID & HeyGen
- Requires API key (set in script)
- Use direct public image URLs (e.g., Imgur, Dropbox)
- D-ID free plan requires `type: audio` and `audio_url`; paid plan supports `type: text`
- See error messages for troubleshooting

---

## 🗂️ Directory Structure

```
CM/
├── input_media/         # Place input images/videos here
├── output/              # All results and generated videos
├── SadTalker/           # SadTalker models and scripts
├── face_pipeline.py     # Face analysis pipeline
├── talking_video_generator.py  # SadTalker + TTS video generator
├── heygen_video.py      # HeyGen API video generator
├── did_api_test.py      # D-ID API video generator
├── personality_predictor.py    # Personality prediction
├── Text_conversion.py   # Text/audio/image conversion
├── requirements_talking_video.txt
├── requirements_personality.txt
├── WINDOWS_SETUP.md     # Windows troubleshooting
└── README.md            # This file
```

---

## 🛠️ Troubleshooting

- See `WINDOWS_SETUP.md` for setuptools and GPU issues
- For SadTalker, check `SadTalker/README.md` and download models manually if needed
- For API errors, ensure image/audio URLs are direct and public
- For poor video quality, use higher resolution images

---

## 📄 License & Contributing

- Most code is Apache 2.0 or MIT (see individual modules)
- Contributions welcome! Open issues or pull requests for improvements

---

## Credits

- SadTalker: https://github.com/OpenTalker/SadTalker
- HeyGen: https://www.heygen.com/
- D-ID: https://www.d-id.com/
- DeepFace: https://github.com/serengil/deepface
- Gemini, Whisper, CLIP, YOLOv8, EasyOCR, scikit-learn, transformers
