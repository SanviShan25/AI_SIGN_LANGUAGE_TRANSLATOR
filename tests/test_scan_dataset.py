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


def test_fixed_splits_kept_and_other_split_unchanged(tmp_path):
    import json

    from training.train_transformer import load_fixed_splits, make_split

    wlasl, aslc = tmp_path / "landmarks", tmp_path / "asl_citizen_landmarks"
    for c in ("a", "b"):
        _write(wlasl, c, 10)
        _write(aslc, c, 3)
    fixed_map = {f"{c}/{i:03d}": s for c in ("a", "b") for i, s in enumerate(["train", "val", "test"])}
    (aslc / "splits.json").write_text(json.dumps(fixed_map))

    only_files, only_labels, _ = scan_dataset(wlasl, min_samples=1)
    base = make_split(only_files, only_labels, seed=42)

    files, labels, _ = scan_dataset([wlasl, aslc], min_samples=1)
    fixed = load_fixed_splits(files)
    assert len(fixed) == 6 and all(str(aslc) in f for f in fixed)
    split = make_split(files, labels, seed=42, fixed=fixed)

    for part in ("train", "val", "test"):
        wl = [x for x in split[part] if str(wlasl) in x[0]]
        assert wl == base[part]                       # WLASL split identical to WLASL-only run
        ac = [x for x in split[part] if str(aslc) in x[0]]
        assert len(ac) == 2 and all(fixed[f] == part for f, _ in ac)   # official split respected
