"""Live webcam: word-level recognition (one prediction per sign).

    python -m inference.realtime_detection

Sign, then drop your hands out of view for a moment -> the sign is classified.
Keys: Q = quit, C = clear.
"""

import argparse
from pathlib import Path
import time

import cv2

from core.config import CFG, CHECKPOINT_PATH
from core.landmarks import LandmarkExtractor, draw_landmarks
from core.model import Predictor
from core.recognizer import StreamingRecognizer


def put(frame, text, y, color=(255, 255, 255), scale=0.8):
    cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", type=Path, default=CHECKPOINT_PATH,
                    help="checkpoint to load, e.g. models/combined.pt")
    ap.add_argument("--threshold", type=float, default=CFG.inference.confidence_threshold)
    args = ap.parse_args()
    CFG.inference.confidence_threshold = args.threshold

    predictor = Predictor(args.model)
    rec = StreamingRecognizer(predictor.predict, CFG.inference)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Webcam not found")
    print(f"{len(predictor.classes)} classes loaded. Q = quit")

    last_msg, fps, t_prev = "", 0.0, time.time()
    with LandmarkExtractor() as ex:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            vec, dbg = ex.process(frame)
            for ev in rec.push(vec):
                if ev.kind == "word":
                    last_msg = f"{ev.label.upper()}  ({ev.confidence:.0%})"
                elif ev.kind == "rejected":
                    last_msg = f"not sure... ({ev.label}? {ev.confidence:.0%})"
                elif ev.kind == "sentence_end":
                    rec.words.clear()

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now

            show = cv2.flip(draw_landmarks(frame, dbg), 1)
            put(show, f"{rec.state.upper()}   {fps:.0f} fps", 30, (0, 255, 255) if rec.in_sign else (200, 200, 200), 0.6)
            put(show, last_msg, 70, (0, 255, 0), 1.1)
            for i, (lbl, p) in enumerate(rec.last_top_k[:3]):
                put(show, f"{lbl}: {p:.0%}", 110 + 28 * i, scale=0.6)
            cv2.imshow("Sign recognition", show)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                rec.reset()
                last_msg = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
