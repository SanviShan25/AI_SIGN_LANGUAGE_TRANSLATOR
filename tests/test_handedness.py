"""Hand -> LEFT/RIGHT slot assignment in LandmarkExtractor._put_hand (no camera needed)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import LEFT_SLICE, POSE_SLICE, RAW_DIM, RIGHT_SLICE  # noqa: E402
from core.landmarks import LandmarkExtractor, _hand_array, _pose_array  # noqa: E402

put_hand = LandmarkExtractor._put_hand


def fake_hand(wrist_x, wrist_y):
    """21 fake MediaPipe landmarks (objects with .x .y .z), landmark 0 at the wrist."""
    return [SimpleNamespace(x=wrist_x + 0.01 * i, y=wrist_y - 0.01 * i, z=0.0) for i in range(21)]


def fake_pose(left_wrist, right_wrist):
    """33 fake MediaPipe pose landmarks; only the ids in POSE_IDS matter."""
    pts = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(33)]
    pts[15] = SimpleNamespace(x=left_wrist[0], y=left_wrist[1], z=0.0)   # left wrist
    pts[16] = SimpleNamespace(x=right_wrist[0], y=right_wrist[1], z=0.0)  # right wrist
    return pts


def frame_with_pose(left_wrist=(0.7, 0.6), right_wrist=(0.3, 0.6)):
    vec = np.zeros(RAW_DIM, np.float32)
    vec[POSE_SLICE] = _pose_array(fake_pose(left_wrist, right_wrist))
    return vec, {"hands": [], "pose": None}


def test_pose_wrist_beats_wrong_label():
    """A mirrored image gives the wrong label; the nearest pose wrist must win."""
    vec, dbg = frame_with_pose()
    hand = _hand_array(fake_hand(0.69, 0.58))  # next to the LEFT wrist
    put_hand(vec, "Right", hand, dbg)
    assert np.allclose(vec[LEFT_SLICE], hand)
    assert not np.any(vec[RIGHT_SLICE])
    assert len(dbg["hands"]) == 1


def test_two_hands_same_label_assigned_by_wrist_in_any_order():
    left = _hand_array(fake_hand(0.72, 0.62))
    right = _hand_array(fake_hand(0.28, 0.61))
    for order in ([left, right], [right, left]):
        vec, dbg = frame_with_pose()
        for h in order:
            put_hand(vec, "Left", h, dbg)
        assert np.allclose(vec[LEFT_SLICE], left)
        assert np.allclose(vec[RIGHT_SLICE], right)


def test_both_hands_nearest_same_wrist_uses_best_pairing():
    """Both hands are closer to the left wrist; the closer one keeps it, whatever the order."""
    near = _hand_array(fake_hand(0.70, 0.60))   # on the left wrist
    far = _hand_array(fake_hand(0.55, 0.60))    # between the wrists, still nearer the left
    for order in ([far, near], [near, far]):
        vec, dbg = frame_with_pose()
        for h in order:
            put_hand(vec, "Right", h, dbg)
        assert np.allclose(vec[LEFT_SLICE], near)
        assert np.allclose(vec[RIGHT_SLICE], far)


def test_fallback_to_label_without_pose():
    vec = np.zeros(RAW_DIM, np.float32)
    dbg = {"hands": [], "pose": None}
    a = _hand_array(fake_hand(0.7, 0.6))
    put_hand(vec, "Right", a, dbg)
    assert np.allclose(vec[RIGHT_SLICE], a) and not np.any(vec[LEFT_SLICE])
    # second hand with the same label goes to the free slot
    b = _hand_array(fake_hand(0.3, 0.6))
    put_hand(vec, "Right", b, dbg)
    assert np.allclose(vec[LEFT_SLICE], b) and np.allclose(vec[RIGHT_SLICE], a)
