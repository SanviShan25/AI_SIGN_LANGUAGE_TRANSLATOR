"""Generate a fake landmark dataset to smoke-test training/evaluation end to end
without any videos or webcam (each class = a different hand trajectory).

    python -m scripts.make_synthetic_data --out datasets/synthetic
    python -m training.train_transformer --data datasets/synthetic --epochs 15
"""

import argparse
from pathlib import Path

import numpy as np

from core.config import HAND_POINTS, POSE_SLICE, RIGHT_SLICE


def fake_clip(cls: int, rng: np.random.Generator) -> np.ndarray:
    t = rng.integers(30, 70)
    raw = np.zeros((t, POSE_SLICE.stop), np.float32)
    # pose: nose, L/R shoulder, L/R elbow, L/R wrist
    pose = np.array([[.5, .3, 0], [.6, .5, 0], [.4, .5, 0], [.65, .65, 0],
                     [.35, .65, 0], [.62, .75, 0], [.38, .55, 0]], np.float32)
    raw[:, POSE_SLICE] = (pose + rng.normal(0, .005, pose.shape)).reshape(-1)
    phase = np.linspace(0, 2 * np.pi, t)
    shape = rng.normal(0, .02, (HAND_POINTS, 3)) + np.linspace(0, .08, HAND_POINTS)[:, None] * [1, -1, 0]
    path = np.stack([.4 + .1 * np.cos(phase * (1 + cls % 3) + cls),
                     .45 + .1 * np.sin(phase * (1 + cls // 3)), np.zeros(t)], 1)
    hand = path[:, None, :] + shape[None] + rng.normal(0, .004, (t, HAND_POINTS, 3))
    raw[:, RIGHT_SLICE] = hand.reshape(t, -1)
    raw[:3], raw[-3:] = 0, 0  # hands not yet visible at the edges
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("datasets/synthetic"))
    ap.add_argument("--classes", type=int, default=6)
    ap.add_argument("--per-class", type=int, default=30)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    for c in range(args.classes):
        d = args.out / f"sign_{c}"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_class):
            np.save(d / f"{i:03d}.npy", fake_clip(c, rng))
    print(f"wrote {args.classes * args.per_class} clips to {args.out}")


if __name__ == "__main__":
    main()
