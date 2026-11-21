"""Turns raw landmarks (T, 147) into model features (seq_len, FEATURE_DIM).

Steps
1. trim frames without hands at start/end
2. normalise position + scale using the shoulders (body-relative, camera-independent)
3. add hand-shape features (each hand relative to its own wrist, scale-free)
4. add hand-present flags
5. resample in time to a fixed length

Pure numpy, so it is shared by training, the webcam app, the dashboard and the API.
"""

from __future__ import annotations

import numpy as np

from core.config import (
    HAND_POINTS, LEFT_SLICE, POSE_MIRROR_PAIRS, POSE_SLICE, RAW_DIM, RIGHT_SLICE,
)

N_POSE = (POSE_SLICE.stop - POSE_SLICE.start) // 3
FEATURE_DIM = RAW_DIM + HAND_POINTS * 3 * 2 + 2  # 147 global + 126 hand-shape + 2 flags = 275


def hands_present(raw: np.ndarray) -> np.ndarray:
    """(T,) bool - at least one hand detected in the frame."""
    return np.any(raw[:, LEFT_SLICE] != 0, axis=1) | np.any(raw[:, RIGHT_SLICE] != 0, axis=1)


def trim(raw: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(hands_present(raw))
    if idx.size == 0:
        return raw[:0]
    return raw[idx[0]: idx[-1] + 1]


def resample(seq: np.ndarray, length: int) -> np.ndarray:
    """Linear interpolation over time to exactly ``length`` frames."""
    t = seq.shape[0]
    if t == length:
        return seq.astype(np.float32)
    if t == 1:
        return np.repeat(seq, length, axis=0).astype(np.float32)
    src = np.linspace(0, t - 1, length)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = (src - lo)[:, None]
    return ((1 - w) * seq[lo] + w * seq[hi]).astype(np.float32)


def _body_frame(raw: np.ndarray):
    """Per-clip centre (3,) and scale () from the shoulders, with fallbacks."""
    pose = raw[:, POSE_SLICE].reshape(len(raw), N_POSE, 3)
    ls, rs = pose[:, 1], pose[:, 2]
    ok = np.any(ls != 0, axis=1) & np.any(rs != 0, axis=1)
    if ok.any():
        centre = ((ls[ok] + rs[ok]) / 2).mean(0)
        scale = np.linalg.norm((ls[ok] - rs[ok])[:, :2], axis=1).mean()
    else:  # no pose -> use the hand points themselves
        pts = raw[:, :RIGHT_SLICE.stop].reshape(-1, 3)
        pts = pts[np.any(pts != 0, axis=1)]
        if len(pts) == 0:
            return np.zeros(3, np.float32), 1.0
        centre = pts.mean(0)
        scale = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) * 2
    centre[2] = 0.0  # z from MediaPipe is already relative
    return centre.astype(np.float32), float(max(scale, 1e-3))


def _hand_shape(hand: np.ndarray) -> np.ndarray:
    """(T, 63) -> wrist-relative, size-normalised (T, 63). Zeros stay zeros."""
    h = hand.reshape(len(hand), HAND_POINTS, 3)
    present = np.any(h != 0, axis=(1, 2))
    rel = h - h[:, :1]
    size = np.linalg.norm(rel[:, 9, :2], axis=1)  # wrist -> middle-finger MCP
    rel = rel / np.maximum(size, 1e-3)[:, None, None]
    rel[~present] = 0
    return rel.reshape(len(hand), -1)


def raw_to_features(raw: np.ndarray, seq_len: int) -> np.ndarray | None:
    """Full pipeline. Returns None if the clip has no usable hand frames."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != RAW_DIM:
        raise ValueError(f"expected (T, {RAW_DIM}) landmarks, got {raw.shape}")
    raw = trim(raw)
    if len(raw) == 0:
        return None

    centre, scale = _body_frame(raw)
    pts = raw.reshape(len(raw), -1, 3)
    mask = np.any(pts != 0, axis=2, keepdims=True)
    glob = np.where(mask, (pts - centre) / scale, 0).reshape(len(raw), -1)

    feats = np.concatenate([
        glob,
        _hand_shape(raw[:, LEFT_SLICE]),
        _hand_shape(raw[:, RIGHT_SLICE]),
        np.any(raw[:, LEFT_SLICE] != 0, axis=1, keepdims=True).astype(np.float32),
        np.any(raw[:, RIGHT_SLICE] != 0, axis=1, keepdims=True).astype(np.float32),
    ], axis=1)
    return resample(feats, seq_len)


# ------------------------------------------------------------- augmentation
def mirror(raw: np.ndarray) -> np.ndarray:
    """Left-handed version of a clip: flip x and swap left/right body parts."""
    out = raw.copy()
    pts = out.reshape(len(out), -1, 3)
    present = np.any(pts != 0, axis=2)
    pts[..., 0] = np.where(present, 1.0 - pts[..., 0], 0)
    out = pts.reshape(len(out), -1)
    left, right = out[:, LEFT_SLICE].copy(), out[:, RIGHT_SLICE].copy()
    out[:, LEFT_SLICE], out[:, RIGHT_SLICE] = right, left
    pose = out[:, POSE_SLICE].reshape(len(out), N_POSE, 3)
    for a, b in POSE_MIRROR_PAIRS:
        pose[:, [a, b]] = pose[:, [b, a]]
    out[:, POSE_SLICE] = pose.reshape(len(out), -1)
    return out


def augment_raw(raw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random speed change, crop, rotation, scale, shift, jitter and mirroring."""
    raw = trim(raw)
    t = len(raw)
    if t < 4:
        return raw
    # speed: 0.8x .. 1.25x
    raw = resample(raw, max(4, int(t * rng.uniform(0.8, 1.25))))
    # random temporal crop keeping >= 85 %
    t = len(raw)
    keep = int(t * rng.uniform(0.85, 1.0))
    start = rng.integers(0, t - keep + 1)
    raw = raw[start:start + keep].copy()

    pts = raw.reshape(len(raw), -1, 3)
    present = np.any(pts != 0, axis=2, keepdims=True)
    c = np.array([0.5, 0.5, 0.0], np.float32)
    ang = np.deg2rad(rng.uniform(-12, 12))
    rot = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]], np.float32)
    s = rng.uniform(0.9, 1.1)
    shift = np.array([*rng.uniform(-0.05, 0.05, 2), 0], np.float32)
    pts = ((pts - c) @ rot.T) * s + c + shift
    pts = pts + rng.normal(0, 0.003, pts.shape).astype(np.float32)
    raw = np.where(present, pts, 0).reshape(len(raw), -1).astype(np.float32)

    if rng.random() < 0.5:
        raw = mirror(raw)
    # occasionally drop a hand for a few frames (simulates tracking loss)
    if rng.random() < 0.3:
        a = rng.integers(0, len(raw))
        raw[a:a + rng.integers(1, 4), LEFT_SLICE if rng.random() < 0.5 else RIGHT_SLICE] = 0
    return raw
