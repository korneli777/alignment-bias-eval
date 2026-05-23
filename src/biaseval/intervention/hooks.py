"""Forward-hook context manager that applies a projection to the residual stream.

Patches one transformer block's output: h' = (h - bias) @ P + bias. The bias
is zero for INLP and the activation mean for LEACE (see `fit_leace`). The
hook resolves the layer via `.model.layers[i]` for Llama/Qwen/Mistral/Gemma
and falls back to `.transformer.h[i]` for GPT-2-style models.
"""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


def _get_layer_module(model: PreTrainedModel, layer_idx: int) -> torch.nn.Module:
    base = getattr(model, "model", None)
    if base is not None and hasattr(base, "layers"):
        return base.layers[layer_idx]
    base = getattr(model, "transformer", None)
    if base is not None and hasattr(base, "h"):
        return base.h[layer_idx]
    raise AttributeError(
        f"Could not locate transformer block list on {type(model).__name__}; "
        "expected `.model.layers` or `.transformer.h`."
    )


class ProjectionHook(AbstractContextManager):
    """Apply a fixed linear projection to one layer's output during forward pass.

    Usage:
        with ProjectionHook(model, P, layer_idx=18, bias=None):
            result = crows_pairs.run(model, tokenizer, spec, ...)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        projection: np.ndarray,
        layer_idx: int,
        *,
        bias: np.ndarray | None = None,
    ) -> None:
        self.model = model
        self.layer_idx = int(layer_idx)
        self.layer = _get_layer_module(model, self.layer_idx)
        param = next(model.parameters())
        self._dtype = param.dtype
        self._device = param.device
        self._P = torch.tensor(projection, dtype=self._dtype, device=self._device)
        self._b = (
            torch.tensor(bias, dtype=self._dtype, device=self._device)
            if bias is not None else None
        )
        self._handle: Any = None

    def _hook(self, _module, _inputs, output):
        # Llama-family blocks return a tuple whose first element is the hidden state.
        if isinstance(output, tuple):
            hs = output[0]
            patched = self._patch(hs)
            return (patched, *output[1:])
        return self._patch(output)

    def _patch(self, hs: torch.Tensor) -> torch.Tensor:
        if self._b is not None:
            return (hs - self._b) @ self._P + self._b
        return hs @ self._P

    def __enter__(self) -> ProjectionHook:
        self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
