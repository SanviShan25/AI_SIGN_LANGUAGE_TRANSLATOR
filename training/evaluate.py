"""Evaluate the trained model on the held-out TEST split.

    python -m training.evaluate

Writes to reports/: metrics.json, classification_report.txt,
confusion_matrix.png, training_curves.png, worst_confusions.txt
"""

import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from core.config import CHECKPOINT_PATH, REPORTS_DIR
from core.model import Predictor


def main():
    split = json.loads((REPORTS_DIR / "split.json").read_text())
    pred = Predictor(CHECKPOINT_PATH)
    classes = pred.classes
    c2i = {c: i for i, c in enumerate(classes)}

    y_true, y_pred, top5 = [], [], []
    for f, label in split["test"]:
        p = pred.predict_proba(np.load(f))
        if p is None:
            continue
        y_true.append(c2i[label])
        y_pred.append(int(p.argmax()))
        top5.append(c2i[label] in np.argsort(p)[::-1][:5])

    metrics = {
        "test_samples": len(y_true),
        "num_classes": len(classes),
        "top1_accuracy": accuracy_score(y_true, y_pred),
        "top5_accuracy": float(np.mean(top5)),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "best_val_acc": pred.metrics.get("val_acc"),
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    report = classification_report(y_true, y_pred, labels=range(len(classes)),
                                   target_names=classes, zero_division=0)
    (REPORTS_DIR / "classification_report.txt").write_text(report)

    pairs = Counter((classes[t], classes[p]) for t, p in zip(y_true, y_pred) if t != p)
    (REPORTS_DIR / "worst_confusions.txt").write_text(
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
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    hist_path = REPORTS_DIR / "history.json"
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
        fig.savefig(REPORTS_DIR / "training_curves.png", dpi=120)
        plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(report)
    print(f"plots and reports saved in {REPORTS_DIR}")


if __name__ == "__main__":
    main()
