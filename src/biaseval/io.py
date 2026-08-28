"""Atomic JSON IO and path conventions for run outputs.

Results live in nested directories per benchmark and per model:
    data/raw_logit_scores/{benchmark}/{model_short_name}__{prompt_mode}.json
    data/probe_results/{model_short_name}/{attribute}.json
    data/intervention_projections/{model_short_name}/{attr}__{method}__L{NN}.npz
    data/intervention_results/{benchmark}/{model_short_name}__{attr}__{prompt_mode}__{method}__L{NN}.json

All writes are atomic (tmp + rename) so partial outputs cannot be mistaken
for completed runs by the resume logic in run_*.py.
"""

from __future__ import annotations

import json
import logging
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import transformers

from biaseval.benchmarks.utils import BenchmarkResult
from biaseval.model_loader import ModelSpec

logger = logging.getLogger(__name__)


def _runtime_metadata() -> dict[str, Any]:
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def logit_result_path(
    results_root: Path, benchmark: str, spec: ModelSpec, prompt_mode: str = "raw",
) -> Path:
    """Score file for one benchmark, model, and prompt condition."""
    return (
        Path(results_root) / "raw_logit_scores" / benchmark
        / f"{spec.short_name}__{prompt_mode}.json"
    )


def probe_result_path(results_root: Path, spec: ModelSpec, attribute: str) -> Path:
    """Per-layer probe accuracies for one model and one attribute."""
    return Path(results_root) / "probe_results" / spec.short_name / f"{attribute}.json"


def activation_dir(results_root: Path, spec: ModelSpec) -> Path:
    """Directory of cached residual-stream activations for one model."""
    return Path(results_root) / "activations" / spec.short_name


def projection_path(
    results_root: Path, spec: ModelSpec, attribute: str, method: str,
    *, layer_idx: int,
) -> Path:
    """Persisted erasure-projection matrix per (model, attribute, method, layer)."""
    return (
        Path(results_root) / "intervention_projections" / spec.short_name
        / f"{attribute}__{method}__L{layer_idx:02d}.npz"
    )


def intervention_result_path(
    results_root: Path, benchmark: str, spec: ModelSpec,
    *, attribute: str, prompt_mode: str, method: str, layer_idx: int,
) -> Path:
    """Score file for one intervened run.

    One model contributes a separate file per attribute, prompt mode, erasure
    method, and layer, so all four appear in the name. The layer index is
    zero-padded to keep a directory listing sorted by depth.
    """
    return (
        Path(results_root) / "intervention_results" / benchmark
        / f"{spec.short_name}__{attribute}__{prompt_mode}__{method}__L{layer_idx:02d}.json"
    )


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomic write — JSON to .tmp, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def _spec_payload(spec: ModelSpec) -> dict[str, Any]:
    return {
        "model_id": spec.model_id,
        "family": spec.family,
        "generation": spec.generation,
        "size": spec.size,
        "variant": spec.variant,
        "num_params": spec.num_params,
        "num_layers": spec.num_layers,
        "hidden_size": spec.hidden_size,
    }


def write_benchmark_result(
    results_root: Path, result: BenchmarkResult, spec: ModelSpec,
) -> Path:
    """Write one benchmark run to its canonical path, and return that path.

    The payload stores the scores alongside the model spec and the library
    versions that produced them. A result file can therefore be traced back to
    its environment without a separate log.
    """
    path = logit_result_path(results_root, result.benchmark, spec, result.prompt_mode)
    write_json(path, {
        "spec": _spec_payload(spec),
        "result": result.to_dict(),
        "runtime": _runtime_metadata(),
    })
    logger.info("Wrote %s", path)
    return path


def write_probe_result(
    results_root: Path, spec: ModelSpec, attribute: str, layer_results: list[dict],
) -> Path:
    """Write one model's per-layer probe accuracies, and return the path.

    Parameter count is dropped from the spec payload: probe results are indexed
    by layer, and model size plays no part in reading them back.
    """
    path = probe_result_path(results_root, spec, attribute)
    spec_payload = _spec_payload(spec)
    spec_payload.pop("num_params", None)  # not relevant for probe payload
    write_json(path, {
        "spec": spec_payload,
        "attribute": attribute,
        "layers": layer_results,
        "runtime": _runtime_metadata(),
    })
    logger.info("Wrote %s", path)
    return path


def write_intervention_result(
    results_root: Path, result: BenchmarkResult, spec: ModelSpec,
    *, attribute: str, method: str, layer_idx: int, sanity: dict[str, Any],
) -> Path:
    """Atomic JSON write for an intervened benchmark run.

    Same payload shape as write_benchmark_result with an extra `intervention`
    block carrying the erasure metadata + sanity-check results.
    """
    path = intervention_result_path(
        results_root, result.benchmark, spec,
        attribute=attribute, prompt_mode=result.prompt_mode,
        method=method, layer_idx=layer_idx,
    )
    write_json(path, {
        "spec": _spec_payload(spec),
        "result": result.to_dict(),
        "intervention": {
            "attribute": attribute,
            "method": method,
            "layer_idx": int(layer_idx),
            "sanity": sanity,
        },
        "runtime": _runtime_metadata(),
    })
    logger.info("Wrote %s", path)
    return path


def is_completed(path: Path) -> bool:
    """Result file exists and parses as valid JSON."""
    if not path.exists():
        return False
    try:
        with open(path) as f:
            json.load(f)
        return True
    except (OSError, json.JSONDecodeError):
        return False
