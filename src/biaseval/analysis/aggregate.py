"""Aggregate per-cell result JSONs into long-format Parquet sheets.

Reads `raw_logit_scores/`, `probe_results/`, `intervention_results/`, and
`activations/direction_*.npy` under `data_root` and writes:
    aggregated/logit.parquet
    aggregated/probe.parquet
    aggregated/intervention.parquet

The figure and analysis layers read only the parquets; the raw JSONs are
needed only when re-aggregating from scratch.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def aggregate_logit_results(data_root: Path) -> pd.DataFrame:
    """Long-format DataFrame: one row per (model, benchmark, prompt_mode, metric)."""
    rows: list[dict] = []
    for path in (Path(data_root) / "raw_logit_scores").glob("*/*.json"):
        with open(path) as f:
            data = json.load(f)
        spec = data["spec"]
        result = data["result"]
        for metric, value in result["summary"].items():
            rows.append({
                "model_id": spec["model_id"],
                "family": spec["family"],
                "generation": spec["generation"],
                "size": spec["size"],
                "variant": spec["variant"],
                "num_params": spec["num_params"],
                "benchmark": result["benchmark"],
                "prompt_mode": result.get("prompt_mode", "raw"),
                "metric": metric,
                "value": value,
            })
    return pd.DataFrame(rows)


def aggregate_probe_results(data_root: Path) -> pd.DataFrame:
    """Long-format DataFrame: one row per (model, attribute, layer)."""
    rows: list[dict] = []
    for path in (Path(data_root) / "probe_results").glob("*/*.json"):
        with open(path) as f:
            data = json.load(f)
        spec = data["spec"]
        for layer in data["layers"]:
            rows.append({
                "model_id": spec["model_id"],
                "family": spec["family"],
                "generation": spec["generation"],
                "size": spec["size"],
                "variant": spec["variant"],
                "attribute": data["attribute"],
                "layer": layer["layer"],
                "layer_normalized": layer["layer_normalized"],
                "mean_accuracy": layer["mean_accuracy"],
                "std_accuracy": layer["std_accuracy"],
            })
    return pd.DataFrame(rows)


def load_probe_directions(data_root: Path) -> dict[tuple[str, str], np.ndarray]:
    """Load every saved direction-vector tensor under data/activations/.

    `data/activations/<short_name>/direction_<attr>.npy` is written by
    train_probes_all_layers(save_directions=True). Each file holds a
    (num_layers, hidden) array of unit-norm mean-difference vectors, one
    per layer.

    Returns a dict keyed by (model_short_name, attribute) so callers can
    pair base/instruct directions for the rotation analysis.
    """
    out: dict[tuple[str, str], np.ndarray] = {}
    base = Path(data_root) / "activations"
    if not base.exists():
        return out
    for model_dir in base.iterdir():
        if not model_dir.is_dir():
            continue
        for npy in model_dir.glob("direction_*.npy"):
            attr = npy.stem.removeprefix("direction_")
            try:
                arr = np.load(npy)
            except (OSError, ValueError) as exc:
                logger.warning("Failed to load %s: %s", npy, exc)
                continue
            out[(model_dir.name, attr)] = arr
    return out


def cross_pair_direction_cosines(
    data_root: Path,
    registry_pairs: list[tuple[str, str, str, str, str]],
    attributes: tuple[str, ...] = ("gender",),
) -> pd.DataFrame:
    """Per layer, cosine similarity between base and instruct direction vectors.

    One row per (pair, attribute, layer). Includes depth_frac so figures can
    plot on a normalised x-axis across models with different layer counts.
    """
    dirs = load_probe_directions(data_root)
    if not dirs:
        return pd.DataFrame()

    rows: list[dict] = []
    for base_id, instruct_id, family, generation, size in registry_pairs:
        base_short = base_id.replace("/", "__")
        inst_short = instruct_id.replace("/", "__")
        for attr in attributes:
            b = dirs.get((base_short, attr))
            i = dirs.get((inst_short, attr))
            if b is None or i is None:
                continue
            n = min(b.shape[0], i.shape[0])
            for layer in range(n):
                bv, iv = b[layer], i[layer]
                nb, ni = float(np.linalg.norm(bv)), float(np.linalg.norm(iv))
                cos = (float(np.dot(bv, iv)) / (nb * ni)) if nb > 0 and ni > 0 else float("nan")
                rows.append({
                    "family": family, "generation": generation, "size": size,
                    "base_id": base_id, "instruct_id": instruct_id,
                    "attribute": attr,
                    "layer": layer,
                    "depth_frac": layer / max(n - 1, 1),
                    "cosine": cos,
                })
    return pd.DataFrame(rows)


def aggregate_intervention_results(data_root: Path) -> pd.DataFrame:
    """Long-format intervention scores: one row per (model, benchmark, attr,
    prompt_mode, method, layer, metric).

    depth_frac = layer_idx / (num_layers - 1) so cross-model layer comparisons
    are meaningful even when models have different depths.
    """
    rows: list[dict] = []
    base = Path(data_root) / "intervention_results"
    if not base.exists():
        return pd.DataFrame()
    for path in base.glob("*/*.json"):
        with open(path) as f:
            data = json.load(f)
        spec = data["spec"]
        result = data["result"]
        intv = data.get("intervention", {})
        sanity = intv.get("sanity", {})
        null = sanity.get("nullification", {})
        ppl  = sanity.get("perplexity", {})
        layer_idx = intv.get("layer_idx")
        n_layers  = spec.get("num_layers")
        depth_frac = (
            layer_idx / max(n_layers - 1, 1)
            if layer_idx is not None and n_layers else None
        )
        for metric, value in result["summary"].items():
            rows.append({
                "model_id": spec["model_id"],
                "family": spec["family"],
                "generation": spec["generation"],
                "size": spec["size"],
                "variant": spec["variant"],
                "num_params": spec["num_params"],
                "num_layers": n_layers,
                "benchmark": result["benchmark"],
                "prompt_mode": result.get("prompt_mode", "raw"),
                "attribute": intv.get("attribute"),
                "method": intv.get("method"),
                "layer_idx": layer_idx,
                "depth_frac": depth_frac,
                "probe_acc_post": null.get("post_intervention_probe_accuracy"),
                "probe_acc_passed": null.get("passed"),
                "perplexity_ratio": ppl.get("ratio"),
                "perplexity_passed": ppl.get("passed"),
                "metric": metric,
                "value": value,
            })
    return pd.DataFrame(rows)


def write_aggregated(data_root: Path, out_dir: Path) -> dict[str, int]:
    """Aggregate everything under data_root and write three parquets to out_dir.

    out_dir is typically `data/aggregated/`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logit_df = aggregate_logit_results(data_root)
    probe_df = aggregate_probe_results(data_root)
    intv_df  = aggregate_intervention_results(data_root)

    if not logit_df.empty:
        logit_df.to_parquet(out_dir / "logit.parquet", index=False)
    if not probe_df.empty:
        probe_df.to_parquet(out_dir / "probe.parquet", index=False)
    if not intv_df.empty:
        intv_df.to_parquet(out_dir / "intervention.parquet", index=False)

    logger.info(
        "Aggregated %d logit rows, %d probe rows, %d intervention rows",
        len(logit_df), len(probe_df), len(intv_df),
    )
    return {
        "logit_rows": len(logit_df),
        "probe_rows": len(probe_df),
        "intervention_rows": len(intv_df),
    }
