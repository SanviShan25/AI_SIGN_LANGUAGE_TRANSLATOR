"""Record your own sign VIDEOS from the webcam (keeps the raw video).

    python -m data_collection.collect_data --sign hello --num 40

Keys: R = record one clip (after a 1 s countdown), Q = quit.
Afterwards: python -m data_collection.extract_landmarks --src datasets/raw_videos

Tip: if you don't need the videos, use record_signs.py instead - it saves
landmarks directly and is much faster.
"""

import argparse
import time
from datetime import datetime

import cv2

from core.config import RAW_VIDEO_DIR


def overlay(frame, lines, color=(0, 255, 0)):
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (20, 40 + 35 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sign", required=True)
    ap.add_argument("--num", type=int, default=40)
    ap.add_argument("--seconds", type=float, default=2.5)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    save_dir = RAW_VIDEO_DIR / args.sign
    save_dir.mkdir(parents=True, exist_ok=True)
    done = len(list(save_dir.glob("*.avi")))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Webcam not found")
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    while done < args.num:
        ok, frame = cap.read()
        if not ok:
            break
        show = cv2.flip(frame, 1)
        overlay(show, [f"Sign: {args.sign}", f"Recorded: {done}/{args.num}", "R = record   Q = quit"])
        cv2.imshow("Data Collection", show)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key != ord("r"):
            continue

        t0 = time.time()
        while time.time() - t0 < 1.0:  # countdown so the clip starts from a neutral pose
            ok, frame = cap.read()
            show = cv2.flip(frame, 1)
            overlay(show, [f"Get ready... {1 - (time.time() - t0):.1f}"], (0, 200, 255))
            cv2.imshow("Data Collection", show)
            cv2.waitKey(1)

        path = save_dir / f"{args.sign}_{datetime.now():%Y%m%d_%H%M%S_%f}.avi"
        out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"XVID"), args.fps, size)
        for _ in range(int(args.seconds * args.fps)):
            ok, frame = cap.read()
            if not ok:
                break
            out.write(frame)  # saved un-mirrored, same as training videos
            show = cv2.flip(frame, 1)
            overlay(show, ["RECORDING"], (0, 0, 255))
            cv2.imshow("Data Collection", show)
            cv2.waitKey(1)
        out.release()
        done += 1
        print("saved", path)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
