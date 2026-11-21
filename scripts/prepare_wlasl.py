"""Organise the Kaggle "WLASL processed" download into the layout extract_landmarks expects.

    python -m scripts.prepare_wlasl --src ~/.cache/kagglehub/datasets/risangbaskoro/wlasl-processed/versions/5

Input  (--src): WLASL_v0.3.json + videos/<video_id>.mp4
Output (--out): datasets/WLASL/dataset/SL/<gloss>/<video_id>.mp4

Videos are SYMLINKED (no 5 GB copy). Only instances whose video file actually
exists are used. If an instance has a real frame range (frame_start > 1 or
frame_end != -1) a small trimmed copy is written instead of a link, unless
--no-trim is given. Re-running is safe: existing outputs are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from core.config import WLASL_DIR


def needs_trim(inst: dict) -> bool:
    return inst.get("frame_start", 1) > 1 or inst.get("frame_end", -1) != -1


def trim_video(src: Path, dst: Path, start: int, end: int) -> bool:
    """Copy frames start..end (1-based, inclusive; end=-1 means last) of src into dst."""
    import cv2

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    idx, written = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx < start:
            continue
        if end != -1 and idx > end:
            break
        out.write(frame)
        written += 1
    cap.release()
    out.release()
    if written == 0:
        dst.unlink(missing_ok=True)
    return written > 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True,
                    help="folder with WLASL_v0.3.json and videos/ (the kagglehub download)")
    ap.add_argument("--out", type=Path, default=WLASL_DIR)
    ap.add_argument("--no-trim", action="store_true", help="symlink everything, ignore frame ranges")
    args = ap.parse_args()

    src = args.src.expanduser().resolve()
    meta, videos = src / "WLASL_v0.3.json", src / "videos"
    if not meta.exists() or not videos.is_dir():
        sys.exit(f"Expected {meta} and {videos}/ - is --src the kagglehub dataset folder?")

    stats: Counter = Counter()
    seen: set[str] = set()
    for entry in json.loads(meta.read_text()):
        gloss = entry["gloss"].strip().replace("/", "_")
        for inst in entry["instances"]:
            vid = str(inst["video_id"])
            video = videos / f"{vid}.mp4"
            if not video.exists():
                stats["missing video"] += 1
                continue
            if vid in seen:  # a video listed under two glosses would leak between classes
                stats["duplicate id skipped"] += 1
                continue
            seen.add(vid)
            dst = args.out / gloss / f"{vid}.mp4"
            if dst.exists() or dst.is_symlink():
                stats["already there"] += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if needs_trim(inst) and not args.no_trim:
                ok = trim_video(video, dst, inst.get("frame_start", 1), inst.get("frame_end", -1))
                stats["trimmed copy" if ok else "trim failed"] += 1
            else:
                dst.symlink_to(video)
                stats["symlinked"] += 1

    classes = [p for p in args.out.iterdir() if p.is_dir()] if args.out.exists() else []
    n_videos = sum(1 for c in classes for _ in c.glob("*.mp4"))
    print(dict(stats))
    print(f"{len(classes)} classes, {n_videos} videos in {args.out}")


if __name__ == "__main__":
    main()
