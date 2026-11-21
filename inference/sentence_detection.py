"""Full live pipeline: webcam -> signs -> gloss -> LLM sentence -> speech.

    python -m inference.sentence_detection
    python -m inference.sentence_detection --no-speech

Sign words one after another (drop hands briefly between signs).
Keep hands down ~1.5 s, or press ENTER, to finish the sentence.
Keys: ENTER = finish sentence, BACKSPACE = delete last word, C = clear, Q = quit.
"""

import argparse
from pathlib import Path
import threading

import cv2

from core.config import CFG, CHECKPOINT_PATH
from core.landmarks import LandmarkExtractor, draw_landmarks
from core.model import Predictor
from core.recognizer import StreamingRecognizer
from inference.realtime_detection import put
from llm_layer.grammar_correction import gloss_to_sentence
from speech.text_to_speech import speak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", type=Path, default=CHECKPOINT_PATH,
                    help="checkpoint to load, e.g. models/combined.pt")
    ap.add_argument("--no-speech", action="store_true")
    args = ap.parse_args()

    predictor = Predictor(args.model)
    rec = StreamingRecognizer(predictor.predict, CFG.inference)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Webcam not found")

    state = {"sentence": "", "backend": "", "busy": False}
    history: list[str] = []

    def finish(words):
        """LLM call in a thread so video never freezes."""
        def job():
            state["busy"] = True
            sentence, backend = gloss_to_sentence(words)
            state.update(sentence=sentence, backend=backend, busy=False)
            history.append(sentence)
            print(f"{' '.join(words)}  ->  {sentence}  [{backend}]")
            if not args.no_speech:
                speak(sentence)
        threading.Thread(target=job, daemon=True).start()

    with LandmarkExtractor() as ex:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            vec, dbg = ex.process(frame)
            for ev in rec.push(vec):
                if ev.kind == "sentence_end" and ev.words:
                    finish(ev.words)

            show = cv2.flip(draw_landmarks(frame, dbg), 1)
            h = show.shape[0]
            cv2.rectangle(show, (0, h - 120), (show.shape[1], h), (0, 0, 0), -1)
            put(show, rec.state.upper(), 30, (0, 255, 255) if rec.in_sign else (180, 180, 180), 0.6)
            if rec.last_top_k:
                lbl, p = rec.last_top_k[0]
                put(show, f"last: {lbl} {p:.0%}", 60, scale=0.6)
            put(show, "Gloss: " + " ".join(w.upper() for w in rec.words), h - 80, (0, 255, 0), 0.8)
            text = "translating..." if state["busy"] else state["sentence"]
            put(show, text, h - 35, (255, 255, 0), 0.9)
            cv2.imshow("Sign -> Sentence", show)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 13 and rec.words:      # ENTER
                finish(rec.end_sentence().words)
            if key == 8 and rec.words:       # BACKSPACE
                rec.words.pop()
            if key == ord("c"):
                rec.reset()
                state["sentence"] = ""

    cap.release()
    cv2.destroyAllWindows()
    if history:
        print("\nConversation:\n" + "\n".join(history))


if __name__ == "__main__":
    main()
