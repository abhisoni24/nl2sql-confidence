"""Config composition: defaults are merged and CLI overrides win."""

from nl2sql_conf.config import load_config


def test_compose_and_override():
    cfg = load_config("configs/experiment/cp1_pilot.yaml",
                      ["data.pairs_per_set=7", "generation.batch_size=1"])
    assert cfg.model.hf_id.startswith("Qwen/")
    assert cfg.data.pairs_per_set == 7
    assert cfg.generation.batch_size == 1
    assert cfg.run_name == "cp1_pilot"
