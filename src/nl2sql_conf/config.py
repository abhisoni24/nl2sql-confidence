"""Layered YAML configuration.

An experiment file lists other config files under ``defaults`` (paths relative to
``configs/``, without ``.yaml``). They are merged in order, then the experiment
file itself, then command-line overrides such as ``generation.batch_size=2``.
Later sources win. This mirrors Hydra's defaults list without the dependency.

Example::

    cfg = load_config("configs/experiment/cp1_pilot.yaml", ["data.pairs_per_set=20"])
    cfg.model.hf_id  # 'Qwen/Qwen2.5-Coder-0.5B-Instruct'
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, OmegaConf

from .utils import REPO_ROOT, repo_path

CONFIG_DIR = REPO_ROOT / "configs"


def load_config(experiment: str | Path, overrides: Sequence[str] = ()) -> DictConfig:
    """Compose an experiment config from its defaults list plus CLI overrides."""
    exp_path = repo_path(experiment)
    exp_cfg = OmegaConf.load(exp_path)

    layers = []
    for name in exp_cfg.pop("defaults", []):
        layer = OmegaConf.load(CONFIG_DIR / f"{name}.yaml")
        layer.pop("defaults", None)   # included files contribute values, not further includes
        layers.append(layer)
    layers.append(exp_cfg)
    layers.append(OmegaConf.from_dotlist(list(overrides)))

    cfg = OmegaConf.merge(*layers)
    # Fail fast on mandatory values (marked ??? in YAML) that were never set.
    missing = OmegaConf.missing_keys(cfg)
    if missing:
        raise ValueError(f"Config has unset mandatory keys: {sorted(missing)}")
    OmegaConf.set_readonly(cfg, True)  # configs are inputs, never mutated by code
    return cfg


def run_dir(cfg: DictConfig) -> Path:
    """Directory holding every output of this run: results/<run_name>/."""
    path = repo_path(cfg.paths.results_root) / cfg.run_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def processed_dir(cfg: DictConfig) -> Path:
    """Directory for this run's sampled pairs and examples."""
    path = repo_path(cfg.paths.processed_dir) / cfg.run_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_resolved_config(cfg: DictConfig, filename: str) -> Path:
    """Write the fully merged config next to the outputs it produced."""
    out = run_dir(cfg) / filename
    OmegaConf.save(cfg, out, resolve=True)
    return out
