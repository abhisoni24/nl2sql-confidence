"""Sequence-level confidence scores from per-token statistics of one greedy decode.

Inputs are aligned per generated SQL token ``t_i``:

* ``logprobs[i]``  = log p(t_i | prompt, t_<i)  (raw model distribution)
* ``entropies[i]`` = H(p(. | prompt, t_<i))     in nats

Worked example: tokens with probabilities [0.9, 0.5, 0.99]
    p_gm        = exp(mean(log p)) = (0.9*0.5*0.99) ** (1/3) ~= 0.764
    p_min       = 0.5
    seq_logprob = log(0.9*0.5*0.99) ~= -0.809
"""

from __future__ import annotations

import math
from typing import Sequence

CONFIDENCE_KEYS = ("p_gm", "p_min", "seq_logprob", "neg_entropy")


def confidence_scores(logprobs: Sequence[float], entropies: Sequence[float]) -> dict:
    """Return every confidence measure; NaN when no SQL tokens were produced."""
    n = len(logprobs)
    if n == 0:
        return {k: float("nan") for k in CONFIDENCE_KEYS} | {"n_tokens": 0}
    mean_lp = sum(logprobs) / n
    return {
        "p_gm": math.exp(mean_lp),                 # geometric-mean token probability
        "p_min": math.exp(min(logprobs)),          # least confident token
        "seq_logprob": float(sum(logprobs)),       # joint log-likelihood (length-biased)
        "neg_entropy": -sum(entropies) / n,        # higher = sharper distributions
        "n_tokens": n,
    }
