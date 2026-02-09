
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

### 5. Media Intelligence API (`app.py`)
- **Face** (InsightFace + DeepFace): detection, embeddings, attributes (emotion/age/gender)
- **Voice** (Whisper + Resemblyzer): speech-to-text, speaker embeddings
- **TTS** (Coqui TTS): text-to-speech with optional voice cloning
- REST API: `POST /face/analyze`, `POST /voice/analyze`, `POST /tts`

---

## 🚀 Installation

**Recommended:** Python 3.8+ and CUDA-compatible GPU for best performance.

Install core dependencies:
```bash
pip install -r requirements_talking_video.txt
pip install -r requirements_personality.txt
pip install insightface mediapipe deepface ultralytics easyocr clip-by-openai opencv-python scikit-learn torch torchvision pillow numpy
```

**Media Intelligence API** (face, voice, TTS):
```bash
pip install -r requirements_media_intelligence.txt
```
- **ffmpeg** must be installed system-wide for audio extraction from video.
- **CPU vs GPU**: Default is CPU. For GPU, install `onnxruntime-gpu` (replace `onnxruntime`) and set `USE_GPU=true` when running.
- First run will download models (InsightFace, Whisper, TTS, etc.).

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

### Media Intelligence API
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
# Or: python app.py
```

**Example curl commands:**

```bash
# Face analysis (image upload)
curl -X POST "http://localhost:8000/face/analyze" \
  -F "file=@Photo1.jpg" \
  -F "save_to_storage=false"

# Voice analysis (audio/video upload)
curl -X POST "http://localhost:8000/voice/analyze" \
  -F "file=@output/Hi.wav" \
  -F "save_to_storage=false"

# TTS (text to speech)
curl -X POST "http://localhost:8000/tts" \
  -F "text=Hello, how are you?" \
  -F "return_base64=false"
```

**Request/Response formats:**

| Endpoint | Request | Response |
|----------|---------|----------|
| `POST /face/analyze` | `file` (image), optional `user_id`, `media_id`, `save_to_storage` | `{ faces: [{bbox, det_score, embedding, attributes}], face_count, embedding_length }` |
| `POST /voice/analyze` | `file` (audio/video), optional `user_id`, `media_id`, `save_to_storage` | `{ text, segments, speaker_embedding, speaker_embedding_dim }` |
| `POST /tts` | `text` (required), optional `speaker_wav` (file), `return_base64` | `{ wav_path }` or `{ audio_base64 }` |

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
├── app.py               # Media Intelligence API (FastAPI)
├── pipeline.py          # Media intelligence pipeline orchestrator
├── config.py            # Config (USE_GPU, etc.)
├── services/            # Media intelligence services
│   ├── face_service.py    # InsightFace + DeepFace
│   ├── asr_service.py     # Whisper ASR
│   ├── speaker_service.py # Resemblyzer speaker embeddings
│   └── tts_service.py     # Coqui TTS
├── utils/
│   └── media_utils.py     # Audio extraction, load_wav, safe_remove
├── storage/
│   └── json_store.py      # JSON persistence (face/speaker/transcript)
├── data/                  # Persisted embeddings & transcripts (created at runtime)
├── tests/
│   └── test_services.py   # Unit tests
├── input_media/         # Place input images/videos here
├── output/              # All results and generated videos
├── SadTalker/           # SadTalker models and scripts
├── face_pipeline.py     # Face analysis pipeline (CLI)
├── talking_video_generator.py  # SadTalker + TTS video generator
├── heygen_video.py      # HeyGen API video generator
├── did_api_test.py      # D-ID API video generator
├── personality_predictor.py    # Personality prediction
├── Text_conversion.py   # Text/audio/image conversion
├── requirements_media_intelligence.txt
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
