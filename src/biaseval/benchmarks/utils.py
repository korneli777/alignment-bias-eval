"""Shared utilities for logit-based benchmark scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from biaseval.benchmarks.framings import FRAMINGS

# raw and instruct are the two baseline conditions; the jb_* framings are the
# recoverability ablation (see benchmarks/framings.py).
PROMPT_MODES = ("raw", "instruct", *FRAMINGS)
COMPLETION_INSTRUCTION = "You are an assistant. Finish this sentence:"
QA_INSTRUCTION = "Answer the following question."


@dataclass
class BenchmarkResult:
    """Output of one benchmark run on one model checkpoint."""

    benchmark: str
    model_id: str
    family: str
    variant: str  # "base" or "instruct"
    prompt_mode: str  # "raw" or "instruct"
    summary: dict[str, float]
    per_example: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def wrap_chat_template(tokenizer: PreTrainedTokenizerBase, user_message: str) -> str:
    """Format `user_message` as a single-turn chat with an open assistant turn.

    Falls back to `"{user_message}\\n\\n"` for tokenizers without a chat
    template, so base and instruct variants stay structurally comparable.
    """
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return f"{user_message}\n\n"


@torch.no_grad()
def sentence_log_prob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sentence: str,
    add_bos: bool = True,
) -> tuple[float, int]:
    """Sum of token log-probs over the whole sentence.

    Returns (sum_log_prob, n_tokens) so callers can length-normalise.
    `add_bos` forces inclusion of the BOS token for tokenizers that don't add
    one automatically (Llama 3+, Qwen).
    """
    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=add_bos)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model(**inputs)
    logits = outputs.logits  # (1, seq_len, vocab)

    # Predict token i from logits at position i-1
    shift_logits = logits[:, :-1, :].float()
    shift_labels = inputs["input_ids"][:, 1:]

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum().item(), token_log_probs.shape[1]


@torch.no_grad()
def conditional_log_prob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    context: str,
    continuation: str,
    *,
    add_special_tokens: bool = True,
) -> tuple[float, int]:
    """log P(continuation | context); returns (sum_log_prob, n_cont_tokens).

    Pass `add_special_tokens=False` when `context` was produced by
    `apply_chat_template`: that string already contains BOS / role markers,
    and re-adding them shifts the alignment.
    """
    ctx_ids = tokenizer(context, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids
    full_ids = tokenizer(context + continuation, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids

    ctx_len = ctx_ids.shape[1]
    full_ids = full_ids.to(model.device)

    outputs = model(input_ids=full_ids)
    logits = outputs.logits[:, :-1, :].float()
    labels = full_ids[:, 1:]

    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    cont_lp = token_log_probs[0, ctx_len - 1 :]
    return cont_lp.sum().item(), cont_lp.shape[0]
