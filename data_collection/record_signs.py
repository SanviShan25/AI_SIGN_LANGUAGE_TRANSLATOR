"""Record signs straight to landmark .npy files (fastest way to build your own dataset).

    python -m data_collection.record_signs --signs hello thanks yes no please --per-sign 30

Keys: SPACE = record one clip, N = next sign, Q = quit.
Each clip is saved to datasets/own_landmarks/<sign>/<timestamp>.npy (change with --out),
kept apart from the WLASL landmarks so the benchmark stays clean. Train on both with
    python -m training.train_transformer --data datasets/landmarks datasets/own_landmarks
Record with different lighting, distance, clothes and both hands for best results.
"""

import argparse
import time
from datetime import datetime

import cv2
import numpy as np

from pathlib import Path

from core.config import OWN_LANDMARK_DIR
from core.landmarks import LandmarkExtractor, draw_landmarks
from core.preprocessing import hands_present


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signs", nargs="+", required=True)
    ap.add_argument("--per-sign", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OWN_LANDMARK_DIR, help="root folder for <sign>/*.npy")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Webcam not found")

    with LandmarkExtractor() as ex:
        for sign in args.signs:
            folder = args.out / sign
            folder.mkdir(parents=True, exist_ok=True)
            count = len(list(folder.glob("*.npy")))
            skip = False
            while count < args.per_sign and not skip:
                ok, frame = cap.read()
                if not ok:
                    break
                _, dbg = ex.process(frame)
                show = cv2.flip(draw_landmarks(frame.copy(), dbg), 1)
                cv2.putText(show, f"{sign}: {count}/{args.per_sign}  SPACE=rec N=next Q=quit",
                            (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Record signs", show)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                if key == ord("n"):
                    skip = True
                if key != ord(" "):
                    continue

                seq, t0 = [], time.time()
                while time.time() - t0 < args.seconds:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    vec, dbg = ex.process(frame)
                    seq.append(vec)
                    show = cv2.flip(draw_landmarks(frame.copy(), dbg), 1)
                    cv2.putText(show, "RECORDING", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.imshow("Record signs", show)
                    cv2.waitKey(1)
                seq = np.stack(seq)
                if hands_present(seq).sum() < 5:
                    print("no hands seen - clip discarded")
                    continue
                np.save(folder / f"{datetime.now():%Y%m%d_%H%M%S_%f}.npy", seq)
                count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("done")


if __name__ == "__main__":
    main()
