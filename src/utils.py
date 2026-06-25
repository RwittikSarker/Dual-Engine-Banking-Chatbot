"""
utils.py
--------
Shared utility functions used across the project.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a consistently-formatted logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    """Load and return the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return config


# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ──────────────────────────────────────────────
# File helpers
# ──────────────────────────────────────────────

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not already exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    """Serialise *obj* to a JSON file."""
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def load_json(path: str | Path) -> Any:
    """Load and return a JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ──────────────────────────────────────────────
# Label helpers
# ──────────────────────────────────────────────

def save_label_encoder(label2id: dict[str, int], id2label: dict[int, str], path: str | Path) -> None:
    """Persist label encoder mappings."""
    payload = {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}
    save_json(payload, path)


def load_label_encoder(path: str | Path) -> tuple[dict[str, int], dict[int, str]]:
    """Load label encoder mappings from disk."""
    payload = load_json(path)
    label2id: dict[str, int] = payload["label2id"]
    id2label: dict[int, str] = {int(k): v for k, v in payload["id2label"].items()}
    return label2id, id2label
