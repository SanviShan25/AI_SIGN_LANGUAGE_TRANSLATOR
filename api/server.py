"""REST API (for a mobile app, website or another service).

    uvicorn api.server:app --host 0.0.0.0 --port 8000
    SIGN_MODEL_PATH=models/combined.pt uvicorn api.server:app --port 8000   # other checkpoint
    open http://localhost:8000/docs

Endpoints
  GET  /health
  GET  /classes
  POST /translate/video      multipart file -> words + sentence
  POST /translate/landmarks  {"frames": [[147 floats], ...]} -> words + sentence
                             (lets a client run MediaPipe itself, e.g. in the browser)
  POST /translate/gloss      {"words": ["ME", "HAPPY"]} -> sentence
  POST /speech               {"text": "..."} -> audio/mpeg
"""

import tempfile
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from core.config import RAW_DIM, model_path
from core.pipeline import translate_landmarks, translate_video
from llm_layer.grammar_correction import gloss_to_sentence
from speech.text_to_speech import synthesize_mp3

app = FastAPI(title="AI Sign Language Translator", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_predictor = None
_extractor = None
_lock = threading.Lock()  # MediaPipe graphs are not thread-safe
MAX_UPLOAD_MB = 50


def predictor():
    global _predictor
    if _predictor is None:
        from core.model import Predictor
        try:
            _predictor = Predictor(model_path())
        except FileNotFoundError as e:
            raise HTTPException(503, str(e))
    return _predictor


def extractor():
    global _extractor
    if _extractor is None:
        from core.landmarks import LandmarkExtractor
        _extractor = LandmarkExtractor()
    return _extractor


class LandmarkRequest(BaseModel):
    frames: list[list[float]] = Field(..., description=f"T x {RAW_DIM} raw landmarks")
    mode: str = "auto"


class GlossRequest(BaseModel):
    words: list[str]


class SpeechRequest(BaseModel):
    text: str
    lang: str = "en"


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": model_path().exists(), "model_path": str(model_path())}


@app.get("/classes")
def classes():
    return {"classes": predictor().classes}


@app.post("/translate/video")
async def translate_video_ep(file: UploadFile = File(...), mode: str = "auto"):
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"max {MAX_UPLOAD_MB} MB")
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
    try:
        with _lock:
            return translate_video(f.name, predictor(), extractor(), mode)
    finally:
        Path(f.name).unlink(missing_ok=True)


@app.post("/translate/landmarks")
def translate_landmarks_ep(req: LandmarkRequest):
    raw = np.asarray(req.frames, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != RAW_DIM:
        raise HTTPException(422, f"frames must be T x {RAW_DIM}, got {list(raw.shape)}")
    return translate_landmarks(raw, predictor(), req.mode)


@app.post("/translate/gloss")
def translate_gloss_ep(req: GlossRequest):
    sentence, backend = gloss_to_sentence(req.words)
    return {"sentence": sentence, "llm_backend": backend}


@app.post("/speech")
def speech_ep(req: SpeechRequest):
    audio = synthesize_mp3(req.text, req.lang)
    if audio is None:
        raise HTTPException(503, "TTS unavailable (needs gTTS + internet)")
    return Response(audio, media_type="audio/mpeg")
