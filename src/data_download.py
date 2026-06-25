from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ensure_dir, get_logger, load_config, save_json

logger = get_logger(__name__)


def download_banking77(config: dict) -> None:
    """Download Banking77 and persist as JSON to data/."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)

    data_dir = ensure_dir(config["paths"]["data_dir"])
    dataset_name: str = config["dataset"]["name"]

    logger.info("Downloading dataset: %s", dataset_name)
    dataset = load_dataset(dataset_name)

    for split_name, split_data in dataset.items():
        rows = [{"text": ex["text"], "label": ex["label"]} for ex in split_data]
        out_path = data_dir / f"{split_name}.json"
        save_json(rows, out_path)
        logger.info("Saved %d rows → %s", len(rows), out_path)

    # Save label names
    label_names = dataset["train"].features["label"].names
    save_json(label_names, data_dir / "label_names.json")
    logger.info("Saved %d label names → %s", len(label_names), data_dir / "label_names.json")

    logger.info("Dataset download complete.")


if __name__ == "__main__":
    cfg = load_config()
    download_banking77(cfg)
