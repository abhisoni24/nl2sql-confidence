"""Slow checks on the real 0.5B model (CPU). Run with:  RUN_SLOW=1 pytest tests/test_model_cpu.py

1. TokenStatsRecorder log-probs equal those computed from generate(output_logits=True).
2. Hidden-state features do not depend on left padding (batched == alone).
"""

import os

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

pytestmark = pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="set RUN_SLOW=1")

CFG = OmegaConf.create({
    "model": {"hf_id": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
              "revision": "ea3f2471cf1b1f0db85067f1ef93848e38e88c25",
              "dtype": "float32", "device": "cpu"},
    "generation": {"max_new_tokens": 24, "batch_size": 2,
                   "loop_guard": {"enabled": True, "ngram": 10, "repeats": 3}},
    "features": {"enabled": True, "layer_fractions": [0.5, 1.0]},
})
PROMPTS = [
    "<|im_start|>user\nWrite SQL: count rows in table singer. Schema: CREATE TABLE singer(id INT, name TEXT, age INT);<|im_end|>\n<|im_start|>assistant\n```sql\n",
    "<|im_start|>user\nWrite SQL: names of singers older than 30, newest first, from CREATE TABLE singer(id INT, name TEXT, age INT, joined DATE);<|im_end|>\n<|im_start|>assistant\n```sql\n",
]


@pytest.fixture(scope="module")
def gen():
    from nl2sql_conf.model import SQLGenerator
    return SQLGenerator(CFG.model, CFG.generation, "```", CFG.features)


def test_recorder_matches_output_logits(gen):
    from transformers import LogitsProcessorList
    from nl2sql_conf.model import TokenStatsRecorder
    tok = gen.tokenizer
    enc = tok(PROMPTS, return_tensors="pt", padding=True)
    L = enc.input_ids.shape[1]
    rec = TokenStatsRecorder()
    cfg = gen._base_config(do_sample=False, output_logits=True)
    with torch.inference_mode():
        out = gen.model.generate(**enc, generation_config=cfg,
                                 logits_processor=LogitsProcessorList([rec]), tokenizer=tok)
    new = out.sequences[:, L:]
    lp_rec, _ = rec.finalize(new)
    lp_ref = torch.stack([torch.log_softmax(l.float(), -1).gather(1, new[:, t:t + 1]).squeeze(1)
                          for t, l in enumerate(out.logits)], 1)
    assert lp_rec.shape == lp_ref.shape
    assert torch.allclose(lp_rec, lp_ref, atol=1e-5)


def test_features_padding_invariant(gen):
    both = gen.generate(PROMPTS)                 # prompt 0 is left-padded in this batch
    alone = gen.generate(PROMPTS[:1])
    assert both[0].sql == alone[0].sql
    f_b, f_a = both[0].features.astype(np.float32), alone[0].features.astype(np.float32)
    assert f_b.shape == (2, 2, gen.model.config.hidden_size)
    rel = np.abs(f_b - f_a).max() / np.abs(f_a).max()
    assert rel < 1e-2, rel
    assert np.allclose(both[0].token_logprobs, alone[0].token_logprobs, atol=1e-3)
