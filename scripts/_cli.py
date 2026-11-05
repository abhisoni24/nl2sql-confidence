"""Shared command-line handling: ``--config <experiment.yaml> [key=value ...]``."""

from __future__ import annotations

import argparse

from omegaconf import DictConfig

from nl2sql_conf.config import load_config


def parse(description: str, extra_args: callable = None) -> tuple[argparse.Namespace, DictConfig]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="experiment YAML, e.g. "
                        "configs/experiment/cp1_pilot.yaml")
    if extra_args:
        extra_args(parser)
    parser.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. "
                        "data.pairs_per_set=20 generation.batch_size=2")
    args = parser.parse_args()
    return args, load_config(args.config, args.overrides)
