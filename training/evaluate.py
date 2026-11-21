"""Evaluate the trained model on the held-out TEST split.

    python -m training.evaluate
    python -m training.evaluate --model models/combined.pt --reports reports/combined
    # WLASL-only model on the combined test set (WLASL + ASL Citizen test signers):
    python -m training.evaluate --split reports/combined/split.json --reports reports/wlasl_model_on_combined_test

Writes to reports/: metrics.json, classification_report.txt,
confusion_matrix.png, training_curves.png, worst_confusions.txt
metrics.json has overall numbers plus "by_source": one entry per dataset
(WLASL, ASL Citizen, ...), judged from the data folder each test clip comes from.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from core.config import CHECKPOINT_PATH, DATASETS_DIR, REPORTS_DIR
from core.model import Predictor

SOURCE_NAMES = {"landmarks": "WLASL", "asl_citizen_landmarks": "ASL Citizen", "own_landmarks": "own"}


def source_of(path: str) -> str:
    """Dataset a landmark file belongs to: name of its <root>/<class>/<file>.npy root."""
    root = Path(path).parent.parent
    name = root.name if root.parent.resolve() == DATASETS_DIR.resolve() else str(root)
    return SOURCE_NAMES.get(name, name)


def scores(y_true, y_pred, top5) -> dict:
    return {"test_samples": len(y_true),
            "top1_accuracy": accuracy_score(y_true, y_pred),
            "top5_accuracy": float(np.mean(top5)),
            "macro_f1": f1_score(y_true, y_pred, average="macro")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=CHECKPOINT_PATH)
    ap.add_argument("--reports", type=Path, default=REPORTS_DIR,
                    help="folder with split.json/history.json from training; outputs go here too")
    ap.add_argument("--split", type=Path,
                    help="split.json whose test set to use (default: <reports>/split.json)")
    args = ap.parse_args()
    reports = args.reports

    split = json.loads((args.split or reports / "split.json").read_text())
    pred = Predictor(args.model)
    classes = pred.classes
    c2i = {c: i for i, c in enumerate(classes)}

    y_true, y_pred, top5, sources = [], [], [], []
    unknown = Counter()
    for f, label in split["test"]:
        if label not in c2i:  # test set from another model with classes this one never saw
            unknown[label] += 1
            continue
        p = pred.predict_proba(np.load(f))
        if p is None:
            continue
        y_true.append(c2i[label])
        y_pred.append(int(p.argmax()))
        top5.append(c2i[label] in np.argsort(p)[::-1][:5])
        sources.append(source_of(f))

    metrics = {**scores(y_true, y_pred, top5),
               "num_classes": len(classes),
               "best_val_acc": pred.metrics.get("val_acc"),
               "model": str(args.model)}
    if unknown:
        metrics["skipped_unknown_class"] = dict(unknown)
    by_source = {}
    for s in sorted(set(sources)):
        idx = [i for i, x in enumerate(sources) if x == s]
        by_source[s] = scores([y_true[i] for i in idx], [y_pred[i] for i in idx], [top5[i] for i in idx])
    metrics["by_source"] = by_source
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    report = classification_report(y_true, y_pred, labels=range(len(classes)),
                                   target_names=classes, zero_division=0)
    (reports / "classification_report.txt").write_text(report)

    pairs = Counter((classes[t], classes[p]) for t, p in zip(y_true, y_pred) if t != p)
    (reports / "worst_confusions.txt").write_text(
        "\n".join(f"{n:3d}  true={a:<15} predicted={b}" for (a, b), n in pairs.most_common(30)))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)), normalize="true")
    size = max(6, len(classes) * 0.3)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    if len(classes) <= 60:
        ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
        ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (test top-1 = {metrics['top1_accuracy']:.1%})")
    fig.tight_layout()
    fig.savefig(reports / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    hist_path = reports / "history.json"
    if hist_path.exists():
        h = json.loads(hist_path.read_text())
        ep = [r["epoch"] for r in h]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        a1.plot(ep, [r["train_loss"] for r in h], label="train")
        a1.plot(ep, [r["val_loss"] for r in h], label="val")
        a1.set_title("Loss")
        a1.legend()
        a2.plot(ep, [r["train_acc"] for r in h], label="train")
        a2.plot(ep, [r["val_acc"] for r in h], label="val")
        a2.set_title("Accuracy")
        a2.legend()
        fig.tight_layout()
        fig.savefig(reports / "training_curves.png", dpi=120)
        plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(f"\n{'test set':<14}{'clips':>6}{'top-1':>8}{'top-5':>8}{'macro F1':>10}")
    for name, m in [*by_source.items(), ("overall", metrics)]:
        print(f"{name:<14}{m['test_samples']:>6}{m['top1_accuracy']:>8.1%}{m['top5_accuracy']:>8.1%}"
              f"{m['macro_f1']:>10.3f}")
    print(report)
    print(f"plots and reports saved in {reports}")


if __name__ == "__main__":
    main()
