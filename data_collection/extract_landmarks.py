"""Videos -> landmark .npy files (T, 147).

Works for WLASL (default) and your own recordings:

    python -m data_collection.extract_landmarks                       # all WLASL classes
    python -m data_collection.extract_landmarks --classes hello yes no
    python -m data_collection.extract_landmarks --src datasets/raw_videos
    python -m data_collection.extract_landmarks --max-classes 20 --workers 4

Input layout : <src>/<class_name>/*.mp4|avi|mov
Output layout: datasets/landmarks/<class_name>/<video_name>.npy
Already-extracted files are skipped, so you can stop and resume any time.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from core.config import LANDMARK_DIR, WLASL_DIR
from core.preprocessing import hands_present

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

_extractor = None


def _init_worker():
    global _extractor
    from core.landmarks import LandmarkExtractor
    _extractor = LandmarkExtractor()


def _process(job: tuple[str, str]) -> tuple[str, str]:
    from core.landmarks import extract_video
    src, dst = job
    try:
        seq = extract_video(src, _extractor)
        if len(seq) == 0:
            return src, "unreadable"
        if hands_present(seq).sum() < 3:
            return src, "no hands"
        np.save(dst, seq)
        return src, "ok"
    except Exception as e:
        return src, f"error: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=WLASL_DIR)
    ap.add_argument("--out", type=Path, default=LANDMARK_DIR)
    ap.add_argument("--classes", nargs="*", help="only these class folders")
    ap.add_argument("--max-classes", type=int, help="take the N classes with the most videos")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    if not args.src.exists():
        sys.exit(f"Source folder not found: {args.src}")

    classes = args.classes or sorted(p.name for p in args.src.iterdir() if p.is_dir())
    if args.max_classes:
        counts = {c: sum(1 for f in (args.src / c).iterdir() if f.suffix.lower() in VIDEO_EXT)
                  for c in classes if (args.src / c).is_dir()}
        classes = sorted(counts, key=counts.get, reverse=True)[: args.max_classes]

    jobs = []
    for c in classes:
        folder = args.src / c
        if not folder.is_dir():
            print(f"skip {c}: folder not found")
            continue
        (args.out / c).mkdir(parents=True, exist_ok=True)
        for f in sorted(folder.iterdir()):
            dst = args.out / c / (f.stem + ".npy")
            if f.suffix.lower() in VIDEO_EXT and not dst.exists():
                jobs.append((str(f), str(dst)))

    print(f"{len(classes)} classes, {len(jobs)} videos to process")
    if not jobs:
        return

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **_):
            return x

    stats: dict[str, int] = {}
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers, initializer=_init_worker) as pool:
            results = pool.map(_process, jobs, chunksize=4)
            for _, status in tqdm(results, total=len(jobs)):
                stats[status.split(":")[0]] = stats.get(status.split(":")[0], 0) + 1
    else:
        _init_worker()
        for job in tqdm(jobs):
            _, status = _process(job)
            stats[status.split(":")[0]] = stats.get(status.split(":")[0], 0) + 1
            if status.startswith("error"):
                print(job[0], status)
    print("done:", stats)


if __name__ == "__main__":
    main()
