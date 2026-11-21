"""Streaming recogniser: feed one frame of raw landmarks at a time, get events.

Segmentation logic (no fixed sliding window, so one sign = one prediction):
  * a sign STARTS when hands appear
  * a sign ENDS when hands disappear for ``idle_frames_to_end_sign`` frames
    (or it runs longer than ``max_sign_frames``) -> classify that segment
  * a SENTENCE ends after ``idle_frames_to_end_sentence`` frames without hands

``predict_fn`` takes raw (T, 147) and returns [(label, prob), ...] sorted by prob,
which keeps this class testable without torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from core.config import LEFT_SLICE, RIGHT_SLICE, InferenceConfig

PredictFn = Callable[[np.ndarray], list]


@dataclass
class Event:
    kind: str                  # "word" | "rejected" | "sentence_end"
    label: str | None = None
    confidence: float = 0.0
    top_k: list = field(default_factory=list)
    words: list = field(default_factory=list)


class StreamingRecognizer:
    def __init__(self, predict_fn: PredictFn, cfg: InferenceConfig | None = None):
        self.predict_fn = predict_fn
        self.cfg = cfg or InferenceConfig()
        self.reset()

    def reset(self):
        self.segment: list[np.ndarray] = []
        self.idle = 0
        self.in_sign = False
        self.words: list[str] = []
        self.last_top_k: list = []

    @property
    def state(self) -> str:
        return "signing" if self.in_sign else "waiting"

    def push(self, vec: np.ndarray) -> list[Event]:
        events: list[Event] = []
        has_hand = bool(np.any(vec[LEFT_SLICE]) or np.any(vec[RIGHT_SLICE]))

        if has_hand:
            self.idle = 0
            self.in_sign = True
            self.segment.append(vec)
            if len(self.segment) >= self.cfg.max_sign_frames:
                events += self._finish_sign()
            return events

        self.idle += 1
        if self.in_sign:
            self.segment.append(vec)  # keep a few blank frames; trimmed later anyway
            if self.idle >= self.cfg.idle_frames_to_end_sign:
                events += self._finish_sign()
        elif self.words and self.idle >= self.cfg.idle_frames_to_end_sentence:
            events.append(self.end_sentence())
        return events

    def _finish_sign(self) -> list[Event]:
        seg = np.stack(self.segment)
        self.segment, self.in_sign = [], False
        hand_frames = int(np.sum(np.any(seg[:, :RIGHT_SLICE.stop] != 0, axis=1)))
        if hand_frames < self.cfg.min_sign_frames:
            return []
        top = self.predict_fn(seg)
        self.last_top_k = top
        if not top:
            return []
        label, conf = top[0]
        if conf < self.cfg.confidence_threshold:
            return [Event("rejected", label, conf, top)]
        if not self.words or self.words[-1] != label:  # drop accidental repeats
            self.words.append(label)
        return [Event("word", label, conf, top)]

    def end_sentence(self) -> Event:
        ev = Event("sentence_end", words=list(self.words))
        self.words = []
        return ev

    def flush(self) -> list[Event]:
        """Call at the end of a video file."""
        events = self._finish_sign() if self.segment else []
        if self.words:
            events.append(self.end_sentence())
        return events
