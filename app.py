"""
Media Intelligence API - FastAPI backend for face and voice analysis.

Endpoints:
- POST /face/analyze - image upload, returns face detection + embeddings + attributes
- POST /voice/analyze - audio/video upload, returns transcript + speaker embedding
- POST /tts - text form field, returns generated WAV path or base64 audio

Run: uvicorn app:app --reload --port 8000
"""

import base64
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from pipeline import MediaIntelligencePipeline
from storage.json_store import (
    save_face_embedding,
    save_speaker_embedding,
    save_transcript,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Media Intelligence API",
    description="Face detection, embeddings, attributes; speech-to-text; speaker embeddings; TTS",
    version="1.0.0",
)

# Pipeline and output dir (one per process)
pipeline = MediaIntelligencePipeline()
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_upload(upload: UploadFile, suffix: str = "") -> str:
    """Save uploaded file to temp location, return path."""
    ext = Path(upload.filename or "file").suffix or suffix
    fd, path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(upload.file.read())
    return path


@app.get("/")
def root():
    return {"message": "Media Intelligence API", "docs": "/docs"}


@app.post("/face/analyze")
async def face_analyze(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    media_id: Optional[str] = Form(None),
    save_to_storage: bool = Form(False),
):
    """
    Analyze image: detect faces, extract embeddings, analyze attributes (emotion/age/gender).
    """
    if not file.filename:
        raise HTTPException(400, "No file provided")

    path = None
    try:
        path = _save_upload(file, ".jpg")
        result = pipeline.process_image(path)

        if "error" in result and result.get("face_count", 0) == 0:
            return JSONResponse(status_code=200, content=result)

        # Optionally persist to storage
        if save_to_storage and result.get("faces"):
            uid = user_id or "default"
            mid = media_id or str(uuid.uuid4())
            for face in result["faces"]:
                save_face_embedding(
                    user_id=uid,
                    media_id=mid,
                    embedding=face.get("embedding", []),
                    bbox=face.get("bbox", []),
                    score=face.get("det_score", 0.0),
                    extra={"attributes": face.get("attributes", {})},
                )

        return JSONResponse(content=result)
    except Exception as e:
        logger.exception(f"face_analyze error: {e}")
        raise HTTPException(500, str(e))
    finally:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


@app.post("/voice/analyze")
async def voice_analyze(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    media_id: Optional[str] = Form(None),
    save_to_storage: bool = Form(False),
):
    """
    Analyze audio/video: transcribe and extract speaker embedding.
    """
    if not file.filename:
        raise HTTPException(400, "No file provided")

    path = None
    try:
        path = _save_upload(file, ".wav")
        result = pipeline.process_media_for_voice(path)

        if "error" in result and not result.get("text") and not result.get("speaker_embedding"):
            return JSONResponse(status_code=200, content=result)

        # Optionally persist to storage
        if save_to_storage:
            uid = user_id or "default"
            mid = media_id or str(uuid.uuid4())
            if result.get("text") or result.get("segments"):
                save_transcript(
                    user_id=uid,
                    media_id=mid,
                    text=result.get("text", ""),
                    segments=result.get("segments", []),
                )
            if result.get("speaker_embedding"):
                save_speaker_embedding(
                    user_id=uid,
                    media_id=mid,
                    embedding=result["speaker_embedding"],
                )

        return JSONResponse(content=result)
    except Exception as e:
        logger.exception(f"voice_analyze error: {e}")
        raise HTTPException(500, str(e))
    finally:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


@app.post("/tts")
async def tts_generate(
    text: str = Form(...),
    speaker_wav: Optional[UploadFile] = File(None),
    return_base64: bool = Form(False),
):
    """
    Generate speech from text. Optional speaker_wav for voice cloning.
    Returns path to WAV file or base64-encoded audio if return_base64=True.
    """
    if not text or not text.strip():
        raise HTTPException(400, "text cannot be empty")

    speaker_path = None
    out_path = None
    try:
        out_name = f"tts_{uuid.uuid4().hex}.wav"
        out_path = str(OUTPUT_DIR / out_name)

        if speaker_wav and speaker_wav.filename:
            speaker_path = _save_upload(speaker_wav, ".wav")
            pipeline.tts_generate(text, out_path, speaker_wav=speaker_path)
        else:
            pipeline.tts_generate(text, out_path, speaker_wav=None)

        if not os.path.exists(out_path):
            raise HTTPException(500, "TTS output file not created")

        if return_base64:
            with open(out_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            return JSONResponse(content={"audio_base64": audio_b64, "format": "wav"})
        else:
            return JSONResponse(content={"wav_path": out_path, "format": "wav"})
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception(f"tts error: {e}")
        raise HTTPException(500, str(e))
    finally:
        if speaker_path and os.path.exists(speaker_path):
            try:
                os.remove(speaker_path)
            except OSError:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
