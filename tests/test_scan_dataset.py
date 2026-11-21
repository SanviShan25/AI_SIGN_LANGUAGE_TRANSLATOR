"""training.train_transformer.scan_dataset with one or several data folders."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("torch")
from scripts.make_synthetic_data import fake_clip  # noqa: E402
from training.train_transformer import scan_dataset  # noqa: E402

RNG = np.random.default_rng(0)


def _write(root: Path, cls: str, n: int, seed_cls: int = 0):
    (root / cls).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        np.save(root / cls / f"{i:03d}.npy", fake_clip(seed_cls, RNG))


def test_single_folder(tmp_path):
    _write(tmp_path / "a", "hello", 3)
    _write(tmp_path / "a", "yes", 3)
    files, labels, classes = scan_dataset(tmp_path / "a", min_samples=2)
    assert classes == ["hello", "yes"] and len(files) == 6


def test_multiple_folders_merge_same_class(tmp_path):
    wlasl, own = tmp_path / "landmarks", tmp_path / "own_landmarks"
    _write(wlasl, "hello", 2)
    _write(wlasl, "yes", 3)
    _write(own, "hello", 2)      # same class name -> merged with WLASL "hello"
    _write(own, "thanks", 1)     # too few clips -> skipped
    files, labels, classes = scan_dataset([wlasl, own], min_samples=3)
    assert classes == ["hello", "yes"]
    assert labels.count("hello") == 4
    assert sum(str(own) in f for f in files) == 2


def test_missing_folder_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        scan_dataset([tmp_path / "nope"], min_samples=1)
