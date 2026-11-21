"""Central configuration. Every script reads paths and hyper-parameters from here,
so training and inference can never silently disagree."""

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASETS_DIR = ROOT / "datasets"
RAW_VIDEO_DIR = DATASETS_DIR / "raw_videos"          # your own webcam recordings
WLASL_DIR = DATASETS_DIR / "WLASL" / "dataset" / "SL"  # WLASL videos: <class>/<id>.mp4
LANDMARK_DIR = DATASETS_DIR / "landmarks"            # extracted (T, RAW_DIM) .npy files (WLASL)
OWN_LANDMARK_DIR = DATASETS_DIR / "own_landmarks"    # your own webcam recordings (record_signs)

MODELS_DIR = ROOT / "models"
MEDIAPIPE_DIR = MODELS_DIR / "mediapipe"
CHECKPOINT_PATH = MODELS_DIR / "sign_transformer.pt"
MODEL_PATH_ENV = "SIGN_MODEL_PATH"


def model_path() -> Path:
    """Checkpoint used by the dashboard and the API: $SIGN_MODEL_PATH if set
    (relative paths are taken from the project root), else CHECKPOINT_PATH."""
    value = os.getenv(MODEL_PATH_ENV, "").strip()
    if not value:
        return CHECKPOINT_PATH
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p

REPORTS_DIR = ROOT / "reports"

# ---------------------------------------------------------------------------
# Raw landmark layout stored on disk (per frame)
#   left hand  : 21 x 3
#   right hand : 21 x 3
#   upper pose : len(POSE_IDS) x 3
# ---------------------------------------------------------------------------
HAND_POINTS = 21
# MediaPipe pose ids: nose, L/R shoulder, L/R elbow, L/R wrist
POSE_IDS = [0, 11, 12, 13, 14, 15, 16]
# pairs swapped when we mirror an example (indices inside POSE_IDS)
POSE_MIRROR_PAIRS = [(1, 2), (3, 4), (5, 6)]

LEFT_SLICE = slice(0, HAND_POINTS * 3)
RIGHT_SLICE = slice(HAND_POINTS * 3, HAND_POINTS * 6)
POSE_SLICE = slice(HAND_POINTS * 6, HAND_POINTS * 6 + len(POSE_IDS) * 3)
RAW_DIM = HAND_POINTS * 6 + len(POSE_IDS) * 3  # 147


@dataclass
class FeatureConfig:
    seq_len: int = 48            # every clip is resampled to this many frames
    min_frames: int = 8          # clips shorter than this (after trimming) are dropped


@dataclass
class ModelConfig:
    d_model: int = 192
    n_heads: int = 4
    n_layers: int = 4
    ff_mult: int = 2
    dropout: float = 0.2


@dataclass
class TrainConfig:
    epochs: int = 150
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    patience: int = 25           # early stopping on val accuracy
    val_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42
    num_workers: int = 0


@dataclass
class InferenceConfig:
    confidence_threshold: float = 0.55
    idle_frames_to_end_sign: int = 6     # hands gone this many frames -> sign finished
    idle_frames_to_end_sentence: int = 45  # ~1.5 s of no hands -> sentence finished
    max_sign_frames: int = 90            # force a prediction if a sign runs this long
    min_sign_frames: int = 8


@dataclass
class Config:
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def to_dict(self):
        return asdict(self)


CFG = Config()
