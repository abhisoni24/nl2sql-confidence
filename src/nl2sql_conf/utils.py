"""Small shared helpers: repository paths, logging, seeding, JSONL I/O."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

# src/nl2sql_conf/utils.py -> repository root is two levels above the package.
REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(relative: str | Path) -> Path:
    """Resolve a config path (relative to the repo root) to an absolute path."""
    p = Path(relative)
    return p if p.is_absolute() else REPO_ROOT / p


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a consistent timestamped format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # HF Hub logs every HTTP request at INFO; keep run logs readable.
    for noisy in ("httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    """Seed Python and NumPy (and torch, if installed) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # analysis scripts do not need torch
        pass


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts (missing file -> empty list)."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Overwrite ``path`` with one JSON object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Append rows and flush, so an interrupted run loses at most one batch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        f.flush()
