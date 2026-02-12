"""
Media Intelligence Pipeline - orchestrates face, ASR, speaker, and TTS services.
Keeps services decoupled and provides a single entrypoint.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from config import USE_GPU
from services.face_service import FaceService
from services.asr_service import ASRService
from services.speaker_service import SpeakerService
from services.tts_service import TTSService

logger = logging.getLogger(__name__)

# Lazy-initialized singletons (one per process)
_face_service: Optional[FaceService] = None
_asr_service: Optional[ASRService] = None
_speaker_service: Optional[SpeakerService] = None
_tts_service: Optional[TTSService] = None


def _get_face_service() -> FaceService:
    global _face_service
    if _face_service is None:
        _face_service = FaceService(use_gpu=USE_GPU)
    return _face_service


def _get_asr_service() -> ASRService:
    global _asr_service
    if _asr_service is None:
        _asr_service = ASRService(use_gpu=USE_GPU)
    return _asr_service


def _get_speaker_service() -> SpeakerService:
    global _speaker_service
    if _speaker_service is None:
        _speaker_service = SpeakerService()
    return _speaker_service


def _get_tts_service() -> TTSService:
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService(use_gpu=USE_GPU)
    return _tts_service


class MediaIntelligencePipeline:
    """
    NostalgiQ-style pipeline: orchestrates face, voice, and TTS services.
    All outputs are JSON-serializable (embeddings as lists of floats).
    """

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process an image: detect faces, extract embeddings, analyze attributes.

        Returns:
            JSON-serializable dict with:
            - faces: list of {bbox, det_score, embedding (list), attributes}
            - face_count: int
            - embedding_length: int (dim of first face's embedding, or 0)
        """
        import cv2

        if not os.path.exists(image_path):
            return {
                "error": f"Image not found: {image_path}",
                "faces": [],
                "face_count": 0,
                "embedding_length": 0,
            }

        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"error": "Failed to load image", "faces": [], "face_count": 0, "embedding_length": 0}

            face_svc = _get_face_service()
            result = face_svc.detect_embed_and_attributes(img)

            faces = result.get("faces", [])
            # Ensure embeddings are lists of floats (JSON-serializable)
            for f in faces:
                emb = f.get("embedding", [])
                if hasattr(emb, "tolist"):
                    f["embedding"] = emb.tolist()
                else:
                    f["embedding"] = [float(x) for x in emb]

            emb_len = len(faces[0]["embedding"]) if faces else 0

            return {
                "faces": faces,
                "face_count": len(faces),
                "embedding_length": emb_len,
            }
        except Exception as e:
            logger.exception(f"process_image failed: {e}")
            return {
                "error": str(e),
                "faces": [],
                "face_count": 0,
                "embedding_length": 0,
            }

    def process_media_for_voice(self, media_path: str) -> Dict[str, Any]:
        """
        Process audio/video: transcribe and extract speaker embedding.

        Returns:
            JSON-serializable dict with:
            - text: transcript text
            - segments: list of {start, end, text}
            - speaker_embedding: list of floats (or null if failed)
            - speaker_embedding_dim: int
        """
        if not os.path.exists(media_path):
            return {
                "error": f"Media not found: {media_path}",
                "text": "",
                "segments": [],
                "speaker_embedding": None,
                "speaker_embedding_dim": 0,
            }

        try:
            asr_svc = _get_asr_service()
            speaker_svc = _get_speaker_service()

            transcript = asr_svc.transcribe(media_path)
            text = transcript.get("text", "")
            segments = transcript.get("segments", [])

            speaker_embedding = None
            try:
                speaker_embedding = speaker_svc.embed_from_media(media_path)
            except Exception as e:
                logger.warning(f"Speaker embedding failed: {e}")

            dim = len(speaker_embedding) if speaker_embedding else 0

            return {
                "text": text,
                "segments": segments,
                "speaker_embedding": speaker_embedding,
                "speaker_embedding_dim": dim,
                "error": transcript.get("error"),
            }
        except Exception as e:
            logger.exception(f"process_media_for_voice failed: {e}")
            return {
                "error": str(e),
                "text": "",
                "segments": [],
                "speaker_embedding": None,
                "speaker_embedding_dim": 0,
            }

    def tts_generate(
        self,
        text: str,
        out_path: str,
        speaker_wav: Optional[str] = None,
    ) -> str:
        """
        Generate speech from text and save to file.

        Args:
            text: Text to synthesize.
            out_path: Output WAV path.
            speaker_wav: Optional reference audio for voice cloning.

        Returns:
            Path to generated WAV file.
        """
        return _get_tts_service().speak_to_file(text, out_path, speaker_wav=speaker_wav)


# Convenience alias
NostalgiQPipeline = MediaIntelligencePipeline
