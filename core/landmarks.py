"""Landmark extraction that works on BOTH MediaPipe APIs.

* mediapipe <= 0.10.2x ships the legacy ``mp.solutions`` API.
* Newer releases (0.10.3x+) removed it and only ship the Tasks API, which needs
  ``.task`` model files -> run ``python -m scripts.download_models`` once.

Output per frame: a float32 vector of length RAW_DIM (147):
    [left hand 21x3 | right hand 21x3 | upper-body pose 7x3]
Missing parts are zeros. Hands are ALWAYS put in the same slot: by the nearest
pose wrist when the body is detected (robust to mirrored images), otherwise by
MediaPipe's handedness label.

Never mirror the frame before calling ``process`` - training videos are not
mirrored, so webcam frames must not be either. Mirror only for display.
"""

from __future__ import annotations

import numpy as np

from core.config import (
    HAND_POINTS, LEFT_SLICE, MEDIAPIPE_DIR, POSE_IDS, POSE_SLICE, RAW_DIM, RIGHT_SLICE,
)

HAND_TASK = MEDIAPIPE_DIR / "hand_landmarker.task"
POSE_TASK = MEDIAPIPE_DIR / "pose_landmarker_lite.task"


def _hand_array(points) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32).reshape(-1)


def _pose_array(points) -> np.ndarray:
    return np.array([[points[i].x, points[i].y, points[i].z] for i in POSE_IDS],
                    dtype=np.float32).reshape(-1)


class LandmarkExtractor:
    """Use as a context manager or call ``close()`` when done.

    >>> with LandmarkExtractor() as ex:
    ...     vec, debug = ex.process(bgr_frame)
    """

    def __init__(self, use_pose: bool = True, min_conf: float = 0.5):
        import mediapipe as mp

        self.use_pose = use_pose
        self.min_conf = min_conf
        self._mp = mp
        self.backend = "solutions" if hasattr(mp, "solutions") else "tasks"
        self._ts_ms = 0
        self._build()

    # ------------------------------------------------------------------ setup
    def _build(self):
        if self.backend == "solutions":
            s = self._mp.solutions
            self._hands = s.hands.Hands(
                static_image_mode=False, max_num_hands=2,
                min_detection_confidence=self.min_conf, min_tracking_confidence=self.min_conf)
            self._pose = s.pose.Pose(
                static_image_mode=False, model_complexity=0,
                min_detection_confidence=self.min_conf,
                min_tracking_confidence=self.min_conf) if self.use_pose else None
            return

        if not HAND_TASK.exists() or (self.use_pose and not POSE_TASK.exists()):
            raise FileNotFoundError(
                "Your mediapipe version uses the Tasks API and needs model files.\n"
                "Run once:  python -m scripts.download_models")

        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision as v

        self._hands = v.HandLandmarker.create_from_options(v.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_TASK)),
            running_mode=v.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=self.min_conf,
            min_hand_presence_confidence=self.min_conf,
            min_tracking_confidence=self.min_conf))
        self._pose = v.PoseLandmarker.create_from_options(v.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(POSE_TASK)),
            running_mode=v.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=self.min_conf,
            min_tracking_confidence=self.min_conf)) if self.use_pose else None

    def reset(self):
        """Call between unrelated videos so tracking state does not leak."""
        if self.backend == "solutions":
            self.close()
            self._build()
        # Tasks API: timestamps keep increasing, a new video just continues.
        self._ts_ms += 1000

    def close(self):
        for obj in (getattr(self, "_hands", None), getattr(self, "_pose", None)):
            if obj is not None:
                obj.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---------------------------------------------------------------- process
    def process(self, bgr_frame: np.ndarray):
        """Returns (vector[RAW_DIM], debug) where debug holds raw points for drawing."""
        import cv2

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        vec = np.zeros(RAW_DIM, dtype=np.float32)
        debug = {"hands": [], "pose": None}

        if self.backend == "solutions":
            rgb.flags.writeable = False
            # pose first: its wrists decide which slot each hand goes into
            if self._pose is not None:
                pr = self._pose.process(rgb)
                if pr.pose_landmarks:
                    vec[POSE_SLICE] = _pose_array(pr.pose_landmarks.landmark)
                    debug["pose"] = vec[POSE_SLICE].reshape(-1, 3)
            hr = self._hands.process(rgb)
            if hr.multi_hand_landmarks:
                for lm, hd in zip(hr.multi_hand_landmarks, hr.multi_handedness):
                    self._put_hand(vec, hd.classification[0].label, _hand_array(lm.landmark), debug)
            return vec, debug

        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        self._ts_ms += 33
        # pose first: its wrists decide which slot each hand goes into
        if self._pose is not None:
            pr = self._pose.detect_for_video(image, self._ts_ms)
            if pr.pose_landmarks:
                vec[POSE_SLICE] = _pose_array(pr.pose_landmarks[0])
                debug["pose"] = vec[POSE_SLICE].reshape(-1, 3)
        hr = self._hands.detect_for_video(image, self._ts_ms)
        for lm, hd in zip(hr.hand_landmarks, hr.handedness):
            self._put_hand(vec, hd[0].category_name, _hand_array(lm), debug)
        return vec, debug

    @staticmethod
    def _put_hand(vec, label, arr, debug):
        """Put one hand (63,) into the LEFT or RIGHT slot of ``vec``.

        MediaPipe's handedness label flips when the image is mirrored, so when the
        pose is already in ``vec`` we instead assign the hand to the pose wrist
        (POSE_IDS index 5 = left, 6 = right) closest to its landmark 0. The label
        is only used as a fallback when no pose was detected.
        """
        wrists = _pose_wrists(vec)
        if wrists is None:
            slot = LEFT_SLICE if label.lower().startswith("left") else RIGHT_SLICE
            # if both detections claim the same side, put the second one in the free slot
            if np.any(vec[slot]):
                slot = _other(slot)
        else:
            d = _wrist_dists(arr, wrists)  # (dist to left wrist, dist to right wrist)
            slot = LEFT_SLICE if d[0] <= d[1] else RIGHT_SLICE
            if np.any(vec[slot]):
                # slot taken: keep whichever pairing of the two hands is closer overall
                old = vec[slot].copy()
                od = _wrist_dists(old, wrists)
                i, j = (0, 1) if slot == LEFT_SLICE else (1, 0)
                if d[i] + od[j] < od[i] + d[j]:
                    vec[_other(slot)] = old          # newcomer wins, move the old hand
                else:
                    slot = _other(slot)              # old hand stays, newcomer takes the free slot
        vec[slot] = arr
        debug["hands"].append(arr.reshape(HAND_POINTS, 3))


def _other(slot: slice) -> slice:
    return RIGHT_SLICE if slot == LEFT_SLICE else LEFT_SLICE


def _pose_wrists(vec: np.ndarray):
    """(left_wrist_xy, right_wrist_xy) from the pose part of ``vec``, or None if no pose."""
    pose = vec[POSE_SLICE].reshape(len(POSE_IDS), 3)
    if not np.any(pose):
        return None
    return pose[5, :2], pose[6, :2]


def _wrist_dists(hand: np.ndarray, wrists) -> tuple[float, float]:
    w0 = hand[:2]  # landmark 0 (wrist) x, y
    return float(np.linalg.norm(w0 - wrists[0])), float(np.linalg.norm(w0 - wrists[1]))


# ---------------------------------------------------------------------- utils
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
              (17, 18), (18, 19), (19, 20)]


def draw_landmarks(frame: np.ndarray, debug: dict) -> np.ndarray:
    """Draw hands + upper body on a BGR frame (in place) without needing mp.solutions."""
    import cv2

    h, w = frame.shape[:2]
    for hand in debug.get("hands", []):
        pts = [(int(x * w), int(y * h)) for x, y, _ in hand]
        for a, b in HAND_EDGES:
            cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)
        for p in pts:
            cv2.circle(frame, p, 3, (255, 255, 255), -1)
    pose = debug.get("pose")
    if pose is not None:
        pts = [(int(x * w), int(y * h)) for x, y, _ in pose]
        for a, b in [(1, 2), (1, 3), (3, 5), (2, 4), (4, 6)]:
            cv2.line(frame, pts[a], pts[b], (120, 255, 120), 2)
    return frame


def extract_video(path, extractor: LandmarkExtractor, max_frames: int | None = None) -> np.ndarray:
    """Read a whole video file -> (T, RAW_DIM) array."""
    import cv2

    extractor.reset()
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(extractor.process(frame)[0])
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, RAW_DIM), np.float32)
