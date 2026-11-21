"""Transformer encoder classifier over landmark sequences + checkpoint helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from core.config import CHECKPOINT_PATH, Config, FeatureConfig, ModelConfig
from core.preprocessing import FEATURE_DIM, raw_to_features


class SignTransformer(nn.Module):
    def __init__(self, num_classes: int, seq_len: int, cfg: ModelConfig, in_dim: int = FEATURE_DIM):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, cfg.d_model), nn.LayerNorm(cfg.d_model), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, cfg.d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos = nn.Parameter(torch.randn(1, seq_len + 1, cfg.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, cfg.d_model * cfg.ff_mult, cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, F)
        x = self.embed(x)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], dim=1)
        x = x + self.pos[:, : x.shape[1]]
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path: Path, model: nn.Module, classes: list[str], cfg: Config, metrics: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": classes,
                "config": cfg.to_dict(), "metrics": metrics}, path)


class Predictor:
    """Loads a checkpoint and predicts from RAW landmarks (T, 147)."""

    def __init__(self, path: Path = CHECKPOINT_PATH, device: torch.device | None = None):
        if not Path(path).exists():
            raise FileNotFoundError(f"No trained model at {path}. Train first: "
                                    "python -m training.train_transformer")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.classes: list[str] = ckpt["classes"]
        self.feat_cfg = FeatureConfig(**ckpt["config"]["features"])
        self.metrics = ckpt.get("metrics", {})
        self.device = device or best_device()
        self.model = SignTransformer(len(self.classes), self.feat_cfg.seq_len,
                                     ModelConfig(**ckpt["config"]["model"]))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_proba(self, raw: np.ndarray) -> np.ndarray | None:
        feats = raw_to_features(raw, self.feat_cfg.seq_len)
        if feats is None:
            return None
        x = torch.from_numpy(feats)[None].to(self.device)
        return torch.softmax(self.model(x), dim=-1)[0].cpu().numpy()

    def predict(self, raw: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
        p = self.predict_proba(raw)
        if p is None:
            return []
        idx = np.argsort(p)[::-1][:top_k]
        return [(self.classes[i], float(p[i])) for i in idx]
