"""SQL generation with per-token confidence statistics and optional hidden-state features.

Design notes
------------
* An explicit ``GenerationConfig`` is built so the checkpoint's own defaults
  (Qwen ships temperature / top-p / repetition penalty) cannot leak in.
* Greedy confidence comes from ``TokenStatsRecorder``, a pass-through logits
  processor. With greedy decoding and no other processors, the scores it sees are
  the raw next-token logits, so it records exactly log p(t_i | prefix) and the
  full-distribution entropy without storing a (batch, steps, vocab) tensor.
  ``tests/test_model_cpu.py`` checks this against ``output_logits``.
* The prompt ends with an opening ```sql fence; decoding stops at the closing
  fence, and only tokens before it are scored (see ``_sql_token_span``).
* ``LoopGuard`` stops a sequence once it starts repeating itself (in CP1, 2.5% of
  decodes looped to the token cap and held up their whole batch).
* ``extract_features`` runs one extra teacher-forced forward pass over
  prompt + generated SQL and keeps a few layers' hidden states (input to the E5 probe).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer, GenerationConfig,
                          LogitsProcessor, LogitsProcessorList, StoppingCriteria,
                          StoppingCriteriaList)

from .utils import get_logger

log = get_logger(__name__)

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass
class Generation:
    sql: str                     # extracted SQL text (fence removed, stripped)
    raw_text: str                # full decoded continuation, for debugging
    token_logprobs: list[float]  # one per scored SQL token
    token_entropies: list[float]
    hit_stop: bool               # True -> closing fence produced
    stop_reason: str             # fence | loop | eos | max_tokens
    n_generated: int = 0         # tokens generated before padding (incl. fence)
    features: np.ndarray | None = field(default=None, repr=False)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Generation hooks
# --------------------------------------------------------------------------- #
class TokenStatsRecorder(LogitsProcessor):
    """Records chosen-token log-prob and entropy at every decoding step.

    At step t the processor sees the logits for token t; the token actually chosen
    at step t only appears in ``input_ids`` at step t+1, so we keep the previous
    step's log-softmax and gather from it then. ``finalize`` handles the last step.
    """

    def __init__(self) -> None:
        self.prev_logp: torch.Tensor | None = None
        self.logprobs: list[torch.Tensor] = []
        self.entropies: list[torch.Tensor] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.prev_logp is not None:
            self.logprobs.append(self.prev_logp.gather(1, input_ids[:, -1:]).squeeze(1))
        logp = torch.log_softmax(scores.float(), dim=-1)
        self.entropies.append(-(logp.exp() * logp).sum(-1))
        self.prev_logp = logp
        return scores  # pass-through: decoding is unchanged

    def finalize(self, new_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (B, T) log-probs and entropies for the T generated tokens."""
        steps = len(self.entropies)
        if steps and len(self.logprobs) < steps:
            last = new_tokens[:, steps - 1:steps]
            self.logprobs.append(self.prev_logp.gather(1, last).squeeze(1))
        lp = torch.stack(self.logprobs, 1).cpu()
        ent = torch.stack(self.entropies, 1).cpu()
        self.prev_logp = None  # free the (B, vocab) tensor
        return lp, ent


def is_looping(tokens: torch.Tensor, ngram: int, repeats: int) -> torch.Tensor:
    """(B,) bool: the trailing ``ngram`` tokens occur at least ``repeats`` times.

    Example (ngram=2, repeats=3): [5, 1, 2, 1, 2, 1, 2] -> True.
    """
    B, T = tokens.shape
    if T < ngram * repeats:
        return torch.zeros(B, dtype=torch.bool, device=tokens.device)
    windows = tokens.unfold(1, ngram, 1)                       # (B, T-ngram+1, ngram)
    tail = tokens[:, -ngram:].unsqueeze(1)                     # (B, 1, ngram)
    hits = (windows == tail).all(-1).sum(-1)                   # occurrences of the tail
    return hits >= repeats


class LoopGuard(StoppingCriteria):
    """Stop sequences whose generated part has fallen into a repetition loop."""

    def __init__(self, prompt_len: int, ngram: int, repeats: int):
        self.prompt_len, self.ngram, self.repeats = prompt_len, ngram, repeats

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> torch.BoolTensor:
        return is_looping(input_ids[:, self.prompt_len:], self.ngram, self.repeats)


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
class SQLGenerator:
    def __init__(self, model_cfg: DictConfig, gen_cfg: DictConfig, stop_string: str,
                 feature_cfg: DictConfig | None = None):
        if model_cfg.get("num_threads"):
            torch.set_num_threads(int(model_cfg.num_threads))
        self.device = pick_device(model_cfg.device)
        self.stop = stop_string
        self.gen_cfg = gen_cfg
        self.feature_cfg = feature_cfg if feature_cfg and feature_cfg.enabled else None

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.hf_id, revision=model_cfg.revision)
        self.tokenizer.padding_side = "left"  # decoder-only batching needs left pad
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_cfg.hf_id, revision=model_cfg.revision,
            dtype=_DTYPES[model_cfg.dtype],
        ).to(self.device).eval()
        self.eos_ids = self.model.generation_config.eos_token_id
        self.special_ids = set(self.tokenizer.all_special_ids)

        self.layers = _decoder_layers(self.model)
        if self.feature_cfg:
            n = len(self.layers)
            # Fractions of depth -> layer indices, e.g. 28 layers, [0.5, 0.75, 1.0] -> [13, 20, 27]
            self.feature_layers = sorted({max(0, round(f * n) - 1)
                                          for f in self.feature_cfg.layer_fractions})
            log.info("Feature layers %s of %d", self.feature_layers, n)
        log.info("Loaded %s on %s (%s)", model_cfg.hf_id, self.device, model_cfg.dtype)

    # ------------------------------------------------------------------ #
    def prompt_length(self, prompt: str) -> int:
        return len(self.tokenizer(prompt).input_ids)

    def _base_config(self, **overrides) -> GenerationConfig:
        g = self.gen_cfg
        return GenerationConfig(
            max_new_tokens=g.max_new_tokens,
            repetition_penalty=1.0,
            stop_strings=[self.stop],
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.eos_ids,
            return_dict_in_generate=True,
            **overrides,
        )

    def _stopping(self, prompt_len: int) -> StoppingCriteriaList:
        lg = self.gen_cfg.loop_guard
        return StoppingCriteriaList(
            [LoopGuard(prompt_len, lg.ngram, lg.repeats)] if lg.enabled else [])

    @torch.inference_mode()
    def generate(self, prompts: list[str]) -> list[Generation]:
        """Greedy decode + confidence statistics (+ features if enabled)."""
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        L = enc.input_ids.shape[1]
        recorder = TokenStatsRecorder()
        out = self.model.generate(
            **enc, generation_config=self._base_config(do_sample=False),
            logits_processor=LogitsProcessorList([recorder]),
            stopping_criteria=self._stopping(L), tokenizer=self.tokenizer)
        new_tokens = out.sequences[:, L:]
        lp, ent = recorder.finalize(new_tokens)

        gens = [self._finalize(new_tokens[b].tolist(), lp[b].tolist(), ent[b].tolist())
                for b in range(len(prompts))]
        if self.feature_cfg:
            feats = self.extract_features(out.sequences, enc.attention_mask, L,
                                          [len(g.token_logprobs) for g in gens])
            for g, f in zip(gens, feats):
                g.features = f
        return gens

    @torch.inference_mode()
    def sample(self, prompts: list[str], k: int, temperature: float, top_p: float,
               seed: int) -> list[list[Generation]]:
        """k sampled SQL strings per prompt (no confidence statistics)."""
        torch.manual_seed(seed)
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        L = enc.input_ids.shape[1]
        out = self.model.generate(
            **enc, generation_config=self._base_config(
                do_sample=True, temperature=temperature, top_p=top_p, top_k=0,
                num_return_sequences=k),
            stopping_criteria=self._stopping(L), tokenizer=self.tokenizer)
        new_tokens = out.sequences[:, L:]                           # (B*k, T)
        flat = [self._finalize(new_tokens[i].tolist(), [], []) for i in range(len(new_tokens))]
        return [flat[i * k:(i + 1) * k] for i in range(len(prompts))]

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def extract_features(self, sequences: torch.Tensor, prompt_mask: torch.Tensor,
                         prompt_len: int, n_sql: list[int]) -> list[np.ndarray]:
        """Hidden states at chosen layers, two positions per layer.

        Returns per example an array (n_layers, 2, hidden) in float16:
          [:, 0] = last prompt token (what the model "knows" before writing SQL)
          [:, 1] = mean over the scored SQL tokens (falls back to [:, 0] if none)
        Later (padding) tokens cannot affect these positions under causal attention.
        """
        captured: dict[int, torch.Tensor] = {}
        hooks = [self.layers[i].register_forward_hook(
            lambda _m, _inp, out, i=i: captured.__setitem__(
                i, out[0] if isinstance(out, tuple) else out))
            for i in self.feature_layers]
        try:
            gen_mask = torch.ones_like(sequences[:, prompt_len:])
            mask = torch.cat([prompt_mask, gen_mask], dim=1)
            pos = (mask.cumsum(-1) - 1).clamp(min=0)   # match generation-time positions
            self.model(input_ids=sequences, attention_mask=mask, position_ids=pos,
                       use_cache=False, logits_to_keep=1)
        finally:
            for h in hooks:
                h.remove()

        feats = []
        for b, n in enumerate(n_sql):
            per_layer = []
            for i in self.feature_layers:
                h = captured[i][b].float()
                last_prompt = h[prompt_len - 1]
                sql_mean = h[prompt_len:prompt_len + n].mean(0) if n else last_prompt
                per_layer.append(torch.stack([last_prompt, sql_mean]))
            feats.append(torch.stack(per_layer).half().cpu().numpy())
        return feats

    # ------------------------------------------------------------------ #
    def _finalize(self, ids: list[int], lps: list[float], ents: list[float]) -> Generation:
        # Cut at the first special token (EOS or the padding of finished rows).
        end, reason = len(ids), "max_tokens"
        for i, tok in enumerate(ids):
            if tok in self.special_ids:
                end, reason = i, "eos"
                break
        ids, lps, ents = ids[:end], lps[:end], ents[:end]

        raw = self.tokenizer.decode(ids)
        n_sql, hit_stop = self._sql_token_span(ids)
        if hit_stop:
            reason = "fence"
        elif self.gen_cfg.loop_guard.enabled and is_looping(
                torch.tensor([ids]), self.gen_cfg.loop_guard.ngram,
                self.gen_cfg.loop_guard.repeats).item():
            reason = "loop"
        sql = raw.split(self.stop, 1)[0].strip()
        return Generation(sql, raw, lps[:n_sql], ents[:n_sql], hit_stop, reason, len(ids))

    def _sql_token_span(self, ids: list[int]) -> tuple[int, bool]:
        """Number of leading tokens that lie entirely before the closing fence.

        Example: tokens ["SELECT", " name", " FROM", " t", "\\n```"] -> 4.
        The last token is the model's decision to stop, not SQL content, so it
        is excluded even though it starts with a newline.
        """
        text = self.tokenizer.decode(ids)
        fence = text.find(self.stop)
        if fence < 0:
            return len(ids), False
        for k in range(len(ids)):
            # Character offset where token k ends = length of decoded prefix incl. k.
            if len(self.tokenizer.decode(ids[:k + 1])) > fence:
                return k, True
        return len(ids), True


def _decoder_layers(model) -> torch.nn.ModuleList:
    """The model's stack of transformer blocks (Qwen2 / Llama / DeepSeek: model.model.layers)."""
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and name.endswith("layers"):
            if best is None or len(mod) > len(best):
                best = mod
    if best is None:
        raise RuntimeError("Could not locate decoder layers for feature extraction")
    return best
