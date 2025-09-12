# Face Analysis Pipeline

A comprehensive Python script that analyzes faces in images and videos, providing identity clustering, age estimation, facial landmarks, scene understanding, object detection, and text extraction.

## Installation

Install the required dependencies:

```bash
pip install insightface mediapipe deepface ultralytics easyocr clip-by-openai opencv-python scikit-learn torch torchvision pillow numpy
```

## Usage

1. Create an `input_media/` folder and place your images/videos there
2. Run the pipeline:

```bash
python face_pipeline.py
```

### Command Line Options

- `--input` or `-i`: Input directory containing media files (default: `input_media`)
- `--output` or `-o`: Output directory for results (default: `output`)

Example:
```bash
python face_pipeline.py --input my_photos --output results
```

## Features

### Face Analysis
- **Face Detection**: Uses InsightFace for accurate face detection and bounding boxes
- **Identity Clustering**: Groups faces by identity using DBSCAN clustering on face embeddings
- **Age Estimation**: Estimates age using DeepFace with DEX model
- **Facial Landmarks**: Detects eyes, nose, and mouth landmarks using MediaPipe FaceMesh

### Scene Understanding
- **Scene Description**: Uses CLIP to generate text descriptions of scenes
- **Object Detection**: Detects objects using YOLOv8
- **Text Extraction**: Extracts visible text using EasyOCR

## Output Structure

```
output/
├── faces/
│   ├── cluster_0/
│   │   ├── image1_face_0.jpg
│   │   └── image2_face_1.jpg
│   ├── cluster_1/
│   │   └── image1_face_1.jpg
│   └── ...
└── metadata.json
```

## Metadata Format

The `metadata.json` file contains comprehensive analysis results:

```json
[
  {
    "filename": "image1.jpg",
    "frame_idx": 0,
    "faces": [
      {
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.95,
        "landmarks": {
          "left_eye": [[x, y], ...],
          "right_eye": [[x, y], ...],
          "nose_tip": [x, y],
          "mouth_corners": [[x, y], [x, y]]
        },
        "age_estimate": 25,
        "cluster_id": 0
      }
    ],
    "scene_text": "a person in an indoor setting",
    "objects_detected": [
      {
        "class": "person",
        "confidence": 0.89,
        "bbox": [x1, y1, x2, y2]
      }
    ],
    "ocr_text": ["Hello World", "Welcome"]
  }
]
```

## Supported Formats

- **Images**: JPG, JPEG, PNG, BMP, TIFF, WEBP
- **Videos**: MP4, AVI, MOV, MKV, WMV, FLV

## Requirements

- Python 3.7+
- CUDA-compatible GPU (optional, for faster processing)
- At least 4GB RAM recommended

## Notes

- The script automatically downloads required models on first run
- Processing time depends on the number of faces and media files
- For videos, the script samples up to 30 frames evenly distributed
- Face clustering uses cosine similarity with DBSCAN algorithm
