"""
data_download.py
----------------
Downloads the Banking77 dataset from the original GitHub repository
and saves it as JSON files in the data/ directory.

Bypasses Hugging Face datasets loading script deprecation errors by fetching the
raw CSV data directly and converting it to the expected JSON schema.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
from pathlib import Path

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ensure_dir, get_logger, load_config, save_json

logger = get_logger(__name__)


def download_banking77(config: dict) -> None:
    """Download Banking77 and persist as JSON to data/."""
    data_dir = ensure_dir(config["paths"]["data_dir"])

    train_url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
    test_url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"

    # 1. Download Train CSV
    logger.info("Downloading training dataset from %s", train_url)
    try:
        with urllib.request.urlopen(train_url) as response:
            train_csv = response.read().decode("utf-8")
    except Exception as e:
        logger.error("Failed to download training data: %s", e)
        sys.exit(1)

    # 2. Download Test CSV
    logger.info("Downloading test dataset from %s", test_url)
    try:
        with urllib.request.urlopen(test_url) as response:
            test_csv = response.read().decode("utf-8")
    except Exception as e:
        logger.error("Failed to download test data: %s", e)
        sys.exit(1)

    # 3. Parse CSV files
    train_rows_raw = list(csv.DictReader(io.StringIO(train_csv)))
    test_rows_raw = list(csv.DictReader(io.StringIO(test_csv)))

    # 4. Extract and sort unique category names for consistent label mapping
    categories = set()
    for r in train_rows_raw:
        categories.add(r["category"])
    for r in test_rows_raw:
        categories.add(r["category"])

    label_names = sorted(list(categories))
    label_to_id = {name: idx for idx, name in enumerate(label_names)}

    logger.info("Detected %d unique intent categories", len(label_names))

    # 5. Format to expected schema: [{"text": "...", "label": label_id}]
    train_rows = [
        {"text": r["text"], "label": label_to_id[r["category"]]}
        for r in train_rows_raw
    ]
    test_rows = [
        {"text": r["text"], "label": label_to_id[r["category"]]}
        for r in test_rows_raw
    ]

    # 6. Persist JSON files
    train_path = data_dir / "train.json"
    test_path = data_dir / "test.json"
    labels_path = data_dir / "label_names.json"

    save_json(train_rows, train_path)
    logger.info("Saved %d training rows → %s", len(train_rows), train_path)

    save_json(test_rows, test_path)
    logger.info("Saved %d test rows → %s", len(test_rows), test_path)

    save_json(label_names, labels_path)
    logger.info("Saved %d label names → %s", len(label_names), labels_path)

    logger.info("Dataset download and preparation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Banking77 dataset")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config YAML (default: configs/config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    download_banking77(cfg)
