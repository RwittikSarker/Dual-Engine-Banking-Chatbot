"""
evaluate.py
-----------
Loads the best trained classifier and evaluates it on the test split.
Generates and saves evaluation artifacts.

Usage:
    python src/evaluate.py
    python src/evaluate.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.utils import ensure_dir, get_logger, load_config, load_json, load_label_encoder

logger = get_logger(__name__)


def evaluate(cfg: dict) -> dict:
    """Evaluate the saved classifier on the test set and return metrics."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    data_dir = Path(cfg["paths"]["data_dir"])
    artifacts_dir = ensure_dir(cfg["paths"]["artifacts_dir"])
    classifier_dir = Path(cfg["paths"]["classifier_dir"]) / "best"
    label_encoder_path = Path(cfg["paths"]["label_encoder"])

    # Validate paths
    if not classifier_dir.exists():
        logger.error("Classifier not found at %s. Run train.py first.", classifier_dir)
        sys.exit(1)
    if not label_encoder_path.exists():
        logger.error("Label encoder not found. Run train.py first.")
        sys.exit(1)

    label2id, id2label = load_label_encoder(label_encoder_path)
    label_names = [id2label[i] for i in range(len(id2label))]

    # Load model
    logger.info("Loading model from %s", classifier_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(classifier_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(classifier_dir))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load test data
    rows = load_json(data_dir / "test.json")
    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]

    # Inference in batches
    batch_size = 32
    all_preds: list[int] = []

    logger.info("Running inference on %d samples…", len(texts))
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=cfg["classifier"]["max_length"],
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
        all_preds.extend(preds)

    y_true = np.array(labels)
    y_pred = np.array(all_preds)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    logger.info("Evaluation results: %s", metrics)

    # Classification report
    report = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)
    report_path = artifacts_dir / "classification_report_final.txt"
    report_path.write_text(report)
    logger.info("Classification report saved → %s", report_path)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", ax=ax, xticklabels=False, yticklabels=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix | Acc={acc:.4f} F1={f1:.4f}")
    plt.tight_layout()
    cm_path = artifacts_dir / "confusion_matrix_final.png"
    fig.savefig(cm_path, dpi=100)
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", cm_path)

    # Per-class metrics bar chart
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    fig2, ax2 = plt.subplots(figsize=(24, 6))
    ax2.bar(range(len(per_class_f1)), sorted(per_class_f1, reverse=True), color="steelblue")
    ax2.set_title("Per-Class F1 Scores (sorted descending)")
    ax2.set_xlabel("Class rank")
    ax2.set_ylabel("F1")
    plt.tight_layout()
    f1_plot_path = artifacts_dir / "per_class_f1.png"
    fig2.savefig(f1_plot_path, dpi=100)
    plt.close(fig2)
    logger.info("Per-class F1 plot saved → %s", f1_plot_path)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Banking77 classifier")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    metrics = evaluate(cfg)
    print("\n=== Final Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")


if __name__ == "__main__":
    main()
