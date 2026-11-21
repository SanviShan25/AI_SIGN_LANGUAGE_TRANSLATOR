"""Download the MediaPipe Tasks model files (needed for mediapipe >= 0.10.30).

    python -m scripts.download_models
"""

import urllib.request

from core.config import MEDIAPIPE_DIR

MODELS = {
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}


def main():
    MEDIAPIPE_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        dst = MEDIAPIPE_DIR / name
        if dst.exists():
            print("exists:", dst)
            continue
        print("downloading", name, "...")
        urllib.request.urlretrieve(url, dst)
        print("saved:", dst, f"({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
