"""Quick check that MediaPipe + your webcam work:  python test_mediapipe.py  (Q to quit)"""
import cv2

from core.landmarks import LandmarkExtractor, draw_landmarks

with LandmarkExtractor() as ex:
    print("MediaPipe backend:", ex.backend)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        vec, dbg = ex.process(frame)
        show = cv2.flip(draw_landmarks(frame, dbg), 1)
        cv2.putText(show, f"hands: {len(dbg['hands'])}  pose: {dbg['pose'] is not None}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("MediaPipe test", show)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
