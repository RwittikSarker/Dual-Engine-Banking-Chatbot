"""
train.py
--------
Fine-tunes DistilBERT on Banking77 for 77-class intent classification.
Logs all parameters, metrics and artefacts to MLflow.
Supports multiple experiment configurations defined in config.yaml.

Usage:
    python src/train.py --config configs/config.yaml
    python src/train.py --config configs/config.yaml --experiment exp_B_lr5e5_bs32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import os

import numpy as np

from src.utils import (
    ensure_dir,
    get_logger,
    load_config,
    load_json,
    save_label_encoder,
    set_seed,
)

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────

def load_split(data_dir: Path, split: str) -> tuple[list[str], list[int]]:
    rows = load_json(data_dir / f"{split}.json")
    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]
    return texts, labels


def build_label_maps(label_names: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    label2id = {name: idx for idx, name in enumerate(label_names)}
    id2label = {idx: name for idx, name in enumerate(label_names)}
    return label2id, id2label


# ──────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────

def build_dataset(texts: list[str], labels: list[int], tokenizer, max_length: int):
    """Tokenise texts and return a torch Dataset."""
    import torch
    from torch.utils.data import Dataset

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

    class IntentDataset(Dataset):
        def __len__(self):
            return len(labels)

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in encodings.items()}
            item["labels"] = torch.tensor(labels[idx], dtype=torch.long)
            return item

    return IntentDataset()


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="weighted", zero_division=0)
    rec = recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ──────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, label_names: list[str], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix (77 classes)", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", out_path)


def save_metrics_plot(history: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (key, title) in zip(axes, [("loss", "Loss"), ("eval_accuracy", "Accuracy")]):
        if key in history:
            ax.plot(history[key], marker="o", label=key)
            ax.set_title(title)
            ax.set_xlabel("Step / Epoch")
            ax.set_ylabel(key)
            ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    logger.info("Metrics plot saved → %s", out_path)


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def run_experiment(cfg: dict, exp_cfg: dict, exp_name: str) -> None:
    """Run a single training experiment and log to MLflow."""
    import mlflow
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from sklearn.metrics import classification_report

    # Seeds
    seed: int = cfg["dataset"]["random_seed"]
    set_seed(seed)

    # Paths
    data_dir = Path(cfg["paths"]["data_dir"])
    models_dir = ensure_dir(cfg["paths"]["models_dir"])
    artifacts_dir = ensure_dir(cfg["paths"]["artifacts_dir"])
    classifier_dir = ensure_dir(models_dir / "classifier")

    # Labels
    label_names: list[str] = load_json(data_dir / "label_names.json")
    label2id, id2label = build_label_maps(label_names)
    save_label_encoder(label2id, id2label, Path(cfg["paths"]["label_encoder"]))

    # Hyper-parameters (experiment can override)
    model_name: str = cfg["classifier"]["model_name"]
    max_length: int = cfg["classifier"]["max_length"]
    lr: float = exp_cfg.get("learning_rate", cfg["classifier"]["learning_rate"])
    batch_size: int = exp_cfg.get("batch_size", cfg["classifier"]["batch_size"])
    epochs: int = exp_cfg.get("epochs", cfg["classifier"]["epochs"])

    logger.info("=== Experiment: %s ===", exp_name)
    logger.info("lr=%s  batch_size=%s  epochs=%s", lr, batch_size, epochs)

    # Tokeniser
    logger.info("Loading tokeniser: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Datasets
    train_texts, train_labels = load_split(data_dir, "train")
    test_texts, test_labels = load_split(data_dir, "test")

    train_dataset = build_dataset(train_texts, train_labels, tokenizer, max_length)
    eval_dataset = build_dataset(test_texts, test_labels, tokenizer, max_length)

    # Model
    logger.info("Loading model: %s", model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
    )

    # MLflow
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=exp_name):
        # Log parameters
        mlflow.log_params({
            "model_name": model_name,
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "max_length": max_length,
            "num_labels": len(label_names),
        })

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(classifier_dir / exp_name),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=lr,
            weight_decay=cfg["classifier"]["weight_decay"],
            warmup_ratio=cfg["classifier"]["warmup_ratio"],
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=str(artifacts_dir / exp_name / "logs"),
            logging_steps=50,
            seed=seed,
            report_to="none",  # We handle MLflow manually
            dataloader_num_workers=0,
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        logger.info("Starting training…")
        train_result = trainer.train()

        # Final evaluation
        logger.info("Running final evaluation…")
        eval_result = trainer.evaluate()

        # Log metrics
        mlflow.log_metrics({
            "train_loss": train_result.training_loss,
            "eval_loss": eval_result.get("eval_loss", 0.0),
            "accuracy": eval_result.get("eval_accuracy", 0.0),
            "precision": eval_result.get("eval_precision", 0.0),
            "recall": eval_result.get("eval_recall", 0.0),
            "f1": eval_result.get("eval_f1", 0.0),
        })

        # Predictions for confusion matrix
        predictions = trainer.predict(eval_dataset)
        y_pred = np.argmax(predictions.predictions, axis=-1)
        y_true = test_labels

        # Confusion matrix
        cm_path = artifacts_dir / f"confusion_matrix_{exp_name}.png"
        save_confusion_matrix(y_true, y_pred, label_names, cm_path)
        mlflow.log_artifact(str(cm_path))

        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=label_names,
            zero_division=0,
        )
        report_path = artifacts_dir / f"classification_report_{exp_name}.txt"
        report_path.write_text(report)
        mlflow.log_artifact(str(report_path))
        logger.info("Classification report:\n%s", report[:500] + "…")

        # Metrics plot
        metrics_plot_path = artifacts_dir / f"metrics_{exp_name}.png"
        # Build a simple history dict from trainer state
        history = {
            "loss": [x["loss"] for x in trainer.state.log_history if "loss" in x],
            "eval_accuracy": [x["eval_accuracy"] for x in trainer.state.log_history if "eval_accuracy" in x],
        }
        save_metrics_plot(history, metrics_plot_path)
        mlflow.log_artifact(str(metrics_plot_path))

        # Save best model (only for first/best experiment)
        best_model_dir = classifier_dir / "best"
        trainer.save_model(str(best_model_dir))
        tokenizer.save_pretrained(str(best_model_dir))
        logger.info("Best model saved → %s", best_model_dir)
        mlflow.log_param("best_model_dir", str(best_model_dir))

        logger.info(
            "Experiment %s complete | acc=%.4f f1=%.4f",
            exp_name,
            eval_result.get("eval_accuracy", 0),
            eval_result.get("eval_f1", 0),
        )


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train DistilBERT on Banking77")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--experiment",
        default=None,
        help="Run a specific experiment by name. If omitted, run all.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Check data exists
    data_dir = Path(cfg["paths"]["data_dir"])
    if not (data_dir / "train.json").exists():
        logger.error("Training data not found. Run: python src/data_download.py")
        sys.exit(1)

    experiments = cfg.get("experiments", [])
    if not experiments:
        # Fallback: single run from main classifier config
        experiments = [{"name": "default", **cfg["classifier"]}]

    for exp in experiments:
        if args.experiment and exp["name"] != args.experiment:
            continue
        run_experiment(cfg, exp, exp["name"])

    logger.info("All experiments complete.")


if __name__ == "__main__":
    main()
