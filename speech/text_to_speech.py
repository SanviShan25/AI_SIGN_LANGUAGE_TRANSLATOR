"""Text -> speech.

* ``speak(text)``        plays on this machine (offline pyttsx3, gTTS fallback), non-blocking
* ``synthesize_mp3(text)`` returns mp3 bytes (gTTS) for the web dashboard / API

    python -m speech.text_to_speech "Hello, how are you?"
"""

from __future__ import annotations

import io
import queue
import sys
import threading

_q: "queue.Queue[str]" = queue.Queue()
_worker: threading.Thread | None = None


def _speak_blocking(text: str):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.say(text)
        engine.runAndWait()
        return
    except Exception as e:
        print(f"[tts] pyttsx3 unavailable ({e}), trying gTTS", file=sys.stderr)
    try:
        import os
        import tempfile
        from gtts import gTTS
        from playsound import playsound  # optional
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            gTTS(text).write_to_fp(f)
        playsound(f.name)
        os.unlink(f.name)
    except Exception as e:
        print(f"[tts] could not speak ({e}). Text was: {text}", file=sys.stderr)


def _loop():
    while True:
        _speak_blocking(_q.get())


def speak(text: str):
    """Queue text to be spoken on a background thread (keeps the webcam loop smooth)."""
    global _worker
    if not text:
        return
    if _worker is None:
        _worker = threading.Thread(target=_loop, daemon=True)
        _worker.start()
    _q.put(text)


def synthesize_mp3(text: str, lang: str = "en") -> bytes | None:
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text, lang=lang).write_to_fp(buf)
        return buf.getvalue()
    except Exception as e:
        print(f"[tts] gTTS failed: {e}", file=sys.stderr)
        return None


if __name__ == "__main__":
    _speak_blocking(" ".join(sys.argv[1:]) or "Sign language translator is ready.")
