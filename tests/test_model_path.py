"""SIGN_MODEL_PATH handling used by the dashboard and the API."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import CHECKPOINT_PATH, MODEL_PATH_ENV, ROOT, model_path  # noqa: E402


def test_default_when_unset(monkeypatch):
    monkeypatch.delenv(MODEL_PATH_ENV, raising=False)
    assert model_path() == CHECKPOINT_PATH


def test_blank_means_default(monkeypatch):
    monkeypatch.setenv(MODEL_PATH_ENV, "  ")
    assert model_path() == CHECKPOINT_PATH


def test_relative_is_from_project_root(monkeypatch):
    monkeypatch.setenv(MODEL_PATH_ENV, "models/combined.pt")
    assert model_path() == ROOT / "models" / "combined.pt"


def test_absolute_is_kept(monkeypatch, tmp_path):
    monkeypatch.setenv(MODEL_PATH_ENV, str(tmp_path / "m.pt"))
    assert model_path() == tmp_path / "m.pt"
