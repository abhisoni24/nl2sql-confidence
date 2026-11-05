"""Prompt construction. All wording comes from ``cfg.prompt``."""

from __future__ import annotations

from omegaconf import DictConfig


def build_prompt(prompt_cfg: DictConfig, tokenizer, schema: str, question: str) -> str:
    """Chat-formatted prompt ending with the pre-filled assistant prefix.

    Example tail of the returned string (Qwen chat format)::

        <|im_start|>assistant
        ```sql
    """
    messages = [
        {"role": "system", "content": prompt_cfg.system},
        {"role": "user", "content": prompt_cfg.user_template.format(
            schema=schema, question=question.strip())},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return text + prompt_cfg.assistant_prefix
