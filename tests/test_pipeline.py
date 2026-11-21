"""Run:  pytest -q"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import CFG, LEFT_SLICE, RAW_DIM, RIGHT_SLICE, InferenceConfig  # noqa: E402
from core.preprocessing import (FEATURE_DIM, augment_raw, mirror, raw_to_features,  # noqa: E402
                                resample, trim)
from core.recognizer import StreamingRecognizer  # noqa: E402
from llm_layer.grammar_correction import rule_based  # noqa: E402
from scripts.make_synthetic_data import fake_clip  # noqa: E402

RNG = np.random.default_rng(0)


def test_features_shape_and_finite():
    raw = fake_clip(1, RNG)
    assert raw.shape[1] == RAW_DIM
    f = raw_to_features(raw, CFG.features.seq_len)
    assert f.shape == (CFG.features.seq_len, FEATURE_DIM)
    assert np.isfinite(f).all()


def test_empty_clip_returns_none():
    assert raw_to_features(np.zeros((20, RAW_DIM), np.float32), 48) is None


def test_trim_and_resample():
    raw = fake_clip(0, RNG)
    assert len(trim(raw)) == len(raw) - 6
    assert resample(raw, 17).shape == (17, RAW_DIM)


def test_translation_invariance():
    """Moving the whole person in the frame must not change the features."""
    raw = fake_clip(2, RNG)
    moved = raw.copy().reshape(len(raw), -1, 3)
    present = np.any(moved != 0, axis=2, keepdims=True)
    moved = np.where(present, moved + [0.1, -0.05, 0], 0).reshape(len(raw), -1)
    a, b = raw_to_features(raw, 48), raw_to_features(moved.astype(np.float32), 48)
    assert np.allclose(a, b, atol=1e-4)


def test_mirror_swaps_hands_and_is_involution():
    raw = fake_clip(3, RNG)
    m = mirror(raw)
    assert np.any(m[:, LEFT_SLICE]) and not np.any(m[:, RIGHT_SLICE])
    assert np.allclose(mirror(m), raw, atol=1e-6)


def test_augment_keeps_valid_shape():
    for _ in range(20):
        a = augment_raw(fake_clip(4, RNG), RNG)
        f = raw_to_features(a, 48)
        assert f is not None and np.isfinite(f).all()


def test_streaming_segments_two_signs():
    calls = []

    def fake_predict(seg):
        calls.append(len(seg))
        return [(f"w{len(calls)}", 0.9)]

    cfg = InferenceConfig(idle_frames_to_end_sign=4, idle_frames_to_end_sentence=10, min_sign_frames=5)
    rec = StreamingRecognizer(fake_predict, cfg)
    blank = np.zeros(RAW_DIM, np.float32)
    events = []
    stream = [*trim(fake_clip(0, RNG)), *[blank] * 6, *trim(fake_clip(1, RNG)), *[blank] * 15]
    for v in stream:
        events += rec.push(v)
    kinds = [e.kind for e in events]
    assert kinds == ["word", "word", "sentence_end"]
    assert events[-1].words == ["w1", "w2"]


def test_low_confidence_is_rejected():
    rec = StreamingRecognizer(lambda s: [("x", 0.2)], InferenceConfig(min_sign_frames=3))
    ev = []
    for v in trim(fake_clip(0, RNG)):
        ev += rec.push(v)
    ev += rec.flush()
    assert ev[0].kind == "rejected" and rec.words == []


def test_rule_based_grammar():
    assert rule_based(["ME", "HAPPY"]) == "I am happy."
    assert rule_based([]) == ""


def test_model_forward():
    torch = pytest.importorskip("torch")
    from core.model import SignTransformer
    m = SignTransformer(5, 48, CFG.model)
    out = m(torch.randn(2, 48, FEATURE_DIM))
    assert out.shape == (2, 5)
