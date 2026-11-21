"""Offline pipeline used by the dashboard and the API: landmarks/video -> sentence."""

from __future__ import annotations

import numpy as np

from core.config import CFG, InferenceConfig
from core.recognizer import StreamingRecognizer
from llm_layer.grammar_correction import gloss_to_sentence


def translate_landmarks(raw: np.ndarray, predictor, mode: str = "auto",
                        cfg: InferenceConfig | None = None) -> dict:
    """mode: 'word' = whole clip is one sign, 'sentence' = segment into signs,
    'auto' = sentence segmentation, falling back to one word if nothing was found."""
    cfg = cfg or CFG.inference
    result = {"words": [], "segments": [], "sentence": "", "llm_backend": "none"}

    if mode in ("sentence", "auto"):
        rec = StreamingRecognizer(predictor.predict, cfg)
        events = []
        for vec in raw:
            events += rec.push(vec)
        events += rec.flush()
        for ev in events:
            if ev.kind in ("word", "rejected"):
                result["segments"].append({"label": ev.label, "confidence": round(ev.confidence, 4),
                                           "accepted": ev.kind == "word",
                                           "top_k": [(l, round(p, 4)) for l, p in ev.top_k]})
            if ev.kind == "sentence_end":
                result["words"] += ev.words

    if mode == "word" or (mode == "auto" and not result["words"]):
        top = predictor.predict(raw, top_k=5)
        if top:
            result["segments"] = [{"label": top[0][0], "confidence": round(top[0][1], 4),
                                   "accepted": True, "top_k": [(l, round(p, 4)) for l, p in top]}]
            result["words"] = [top[0][0]]

    if result["words"]:
        result["sentence"], result["llm_backend"] = gloss_to_sentence(result["words"])
    return result


def translate_video(path, predictor, extractor, mode: str = "auto") -> dict:
    from core.landmarks import extract_video
    raw = extract_video(path, extractor)
    out = translate_landmarks(raw, predictor, mode)
    out["frames"] = int(len(raw))
    return out
