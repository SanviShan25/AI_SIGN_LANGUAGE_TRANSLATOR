"""Train the sign classifier on datasets/landmarks/<class>/*.npy

    python -m training.train_transformer
    python -m training.train_transformer --epochs 80 --min-samples 8

Writes
  models/sign_transformer.pt   best checkpoint (weights + classes + config)
  reports/split.json           exact train/val/test file lists (test is never trained on)
  reports/history.json         loss/accuracy per epoch
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from core.config import CFG, CHECKPOINT_PATH, LANDMARK_DIR, REPORTS_DIR
from core.model import SignTransformer, best_device, save_checkpoint
from core.preprocessing import augment_raw, raw_to_features, trim


# ------------------------------------------------------------------ data
def scan_dataset(root: Path, min_samples: int):
    files, labels = [], []
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        clips = sorted(cdir.glob("*.npy"))
        usable = [f for f in clips if len(trim(np.load(f))) >= CFG.features.min_frames]
        if len(usable) < min_samples:
            print(f"  skip '{cdir.name}': only {len(usable)} usable clips (< {min_samples})")
            continue
        files += [str(f) for f in usable]
        labels += [cdir.name] * len(usable)
    classes = sorted(set(labels))
    return files, labels, classes


def make_split(files, labels, seed):
    tc = CFG.train
    f_tr, f_te, y_tr, y_te = train_test_split(files, labels, test_size=tc.test_size,
                                              stratify=labels, random_state=seed)
    f_tr, f_va, y_tr, y_va = train_test_split(f_tr, y_tr, test_size=tc.val_size / (1 - tc.test_size),
                                              stratify=y_tr, random_state=seed)
    return {"train": list(zip(f_tr, y_tr)), "val": list(zip(f_va, y_va)), "test": list(zip(f_te, y_te))}


class LandmarkDataset(Dataset):
    def __init__(self, items, class_to_idx, augment: bool, seed: int = 0):
        self.raw = [np.load(f).astype(np.float32) for f, _ in items]
        self.y = [class_to_idx[c] for _, c in items]
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.seq_len = CFG.features.seq_len
        # cache the deterministic features for val/test
        self.cache = None if augment else [raw_to_features(r, self.seq_len) for r in self.raw]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        if self.cache is not None:
            x = self.cache[i]
        else:
            x = raw_to_features(augment_raw(self.raw[i], self.rng), self.seq_len)
            if x is None:  # augmentation removed every hand frame - fall back to original
                x = raw_to_features(self.raw[i], self.seq_len)
        return torch.from_numpy(x), self.y[i]


# --------------------------------------------------------------- training
def run_epoch(model, loader, device, criterion, optimizer=None, scheduler=None):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=LANDMARK_DIR)
    ap.add_argument("--out", type=Path, default=CHECKPOINT_PATH)
    ap.add_argument("--epochs", type=int, default=CFG.train.epochs)
    ap.add_argument("--batch-size", type=int, default=CFG.train.batch_size)
    ap.add_argument("--lr", type=float, default=CFG.train.lr)
    ap.add_argument("--min-samples", type=int, default=6, help="drop classes with fewer clips")
    args = ap.parse_args()

    tc = CFG.train
    tc.epochs, tc.batch_size, tc.lr = args.epochs, args.batch_size, args.lr
    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)

    print(f"Scanning {args.data} ...")
    files, labels, classes = scan_dataset(args.data, args.min_samples)
    if len(classes) < 2:
        raise SystemExit("Need at least 2 classes with enough clips. Extract/record more data first.")
    print(f"{len(files)} clips, {len(classes)} classes")

    split = make_split(files, labels, tc.seed)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "split.json").write_text(json.dumps({"classes": classes, **split}, indent=1))
    c2i = {c: i for i, c in enumerate(classes)}

    train_ds = LandmarkDataset(split["train"], c2i, augment=True, seed=tc.seed)
    val_ds = LandmarkDataset(split["val"], c2i, augment=False)
    print(f"train {len(train_ds)} | val {len(val_ds)} | test {len(split['test'])}")

    # balance classes: sample rarer classes more often
    counts = np.bincount(train_ds.y, minlength=len(classes))
    weights = torch.tensor([1.0 / counts[y] for y in train_ds.y], dtype=torch.double)
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
    train_dl = DataLoader(train_ds, tc.batch_size, sampler=sampler, num_workers=tc.num_workers, drop_last=len(train_ds) > tc.batch_size)
    val_dl = DataLoader(val_ds, tc.batch_size * 2, shuffle=False)

    device = best_device()
    model = SignTransformer(len(classes), CFG.features.seq_len, CFG.model).to(device)
    print(f"device={device}  params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss(label_smoothing=tc.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    steps = tc.epochs * len(train_dl)
    warm = max(1, int(0.05 * steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))

    best, bad, history = -1.0, 0, []
    for ep in range(1, tc.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_dl, device, criterion, optimizer, scheduler)
        va_loss, va_acc = run_epoch(model, val_dl, device, criterion)
        history.append(dict(epoch=ep, train_loss=tr_loss, train_acc=tr_acc, val_loss=va_loss, val_acc=va_acc))
        flag = ""
        if va_acc > best:
            best, bad, flag = va_acc, 0, " *saved*"
            save_checkpoint(args.out, model, classes, CFG, {"val_acc": va_acc, "epoch": ep})
        else:
            bad += 1
        print(f"ep {ep:3d} | train {tr_loss:.3f}/{tr_acc:.3f} | val {va_loss:.3f}/{va_acc:.3f} | {time.time() - t0:.1f}s{flag}")
        if bad >= tc.patience:
            print(f"early stop (no val improvement for {tc.patience} epochs)")
            break

    (REPORTS_DIR / "history.json").write_text(json.dumps(history, indent=1))
    print(f"\nbest val acc {best:.3f}  -> {args.out}")
    print("Next: python -m training.evaluate")


if __name__ == "__main__":
    main()
