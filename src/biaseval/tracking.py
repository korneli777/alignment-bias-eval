"""Thin wrapper around Weights & Biases.

If WANDB_API_KEY is unset or `enabled=False`, all tracker calls become
no-ops so the rest of the code can call `tracker.log(...)` unconditionally.
"""

from __future__ import annotations

import logging
import os
from contextlib import AbstractContextManager
from typing import Any

logger = logging.getLogger(__name__)


class _NullRun(AbstractContextManager):
    """Tracker used when W&B is disabled. Every call is a no-op.

    Callers can then log unconditionally, with no `if tracker is not None`
    guard at each call site.
    """

    def log(self, *_: Any, **__: Any) -> None:
        """Discard the metrics."""

    def summary_update(self, *_: Any, **__: Any) -> None:
        """Discard the summary values."""

    def __exit__(self, *_: object) -> None:
        pass


class WandbRun(AbstractContextManager):
    """Tracker backed by a live W&B run, closed on context exit."""

    def __init__(self, run: Any) -> None:
        self._run = run

    def log(self, data: dict, step: int | None = None) -> None:
        """Log one step of metrics to the run."""
        self._run.log(data, step=step)

    def summary_update(self, data: dict) -> None:
        """Write final values onto the run summary, overwriting any earlier ones."""
        for k, v in data.items():
            self._run.summary[k] = v

    def __exit__(self, *_: object) -> None:
        self._run.finish()


def init_run(
    *,
    project: str | None = None,
    name: str | None = None,
    config: dict | None = None,
    tags: list[str] | None = None,
    job_type: str | None = None,
    enabled: bool = True,
) -> WandbRun | _NullRun:
    """Initialize a W&B run, or return a null context if disabled."""
    if not enabled or not os.environ.get("WANDB_API_KEY"):
        if enabled:
            logger.warning("WANDB_API_KEY not set — disabling W&B logging.")
        return _NullRun()

    import wandb

    run = wandb.init(
        project=project or os.environ.get("WANDB_PROJECT", "alignment-bias-eval"),
        entity=os.environ.get("WANDB_ENTITY"),
        name=name,
        config=config or {},
        tags=tags or [],
        job_type=job_type,
        reinit=True,
    )
    return WandbRun(run)
