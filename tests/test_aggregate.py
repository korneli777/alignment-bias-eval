"""Unit tests for conversion from result JSONs to analysis tables."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from biaseval.analysis.aggregate import (
    aggregate_intervention_results,
    aggregate_logit_results,
    aggregate_probe_results,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _spec() -> dict:
    return {
        "model_id": "example/base",
        "family": "example",
        "generation": "Example 1",
        "size": "1B",
        "variant": "base",
        "num_params": 1_000_000_000,
        "num_layers": 5,
    }


def test_aggregate_logit_results_flattens_summary_metrics(tmp_path: Path):
    _write_json(
        tmp_path / "raw_logit_scores" / "bbq" / "result.json",
        {
            "spec": _spec(),
            "result": {
                "benchmark": "bbq",
                "prompt_mode": "raw",
                "summary": {"bias": 0.1, "accuracy": 0.8},
            },
        },
    )

    frame = aggregate_logit_results(tmp_path)

    assert set(frame["metric"]) == {"bias", "accuracy"}
    assert set(frame["value"]) == {0.1, 0.8}
    assert frame["model_id"].unique().tolist() == ["example/base"]
    assert frame["prompt_mode"].unique().tolist() == ["raw"]


def test_aggregate_probe_results_keeps_layer_metadata(tmp_path: Path):
    _write_json(
        tmp_path / "probe_results" / "example__base" / "gender.json",
        {
            "spec": _spec(),
            "attribute": "gender",
            "layers": [
                {
                    "layer": 2,
                    "layer_normalized": 0.5,
                    "mean_accuracy": 0.75,
                    "std_accuracy": 0.04,
                }
            ],
        },
    )

    row = aggregate_probe_results(tmp_path).iloc[0]

    assert row["attribute"] == "gender"
    assert row["layer"] == 2
    assert row["layer_normalized"] == pytest.approx(0.5)
    assert row["mean_accuracy"] == pytest.approx(0.75)


def test_aggregate_interventions_includes_depth_and_sanity_gates(tmp_path: Path):
    _write_json(
        tmp_path / "intervention_results" / "crows_pairs" / "result.json",
        {
            "spec": _spec(),
            "result": {
                "benchmark": "crows_pairs",
                "prompt_mode": "raw",
                "summary": {"overall": 51.0},
            },
            "intervention": {
                "attribute": "gender",
                "method": "inlp",
                "layer_idx": 2,
                "sanity": {
                    "nullification": {
                        "post_intervention_probe_accuracy": 0.51,
                        "passed": True,
                    },
                    "perplexity": {"ratio": 1.1, "passed": True},
                },
            },
        },
    )

    row = aggregate_intervention_results(tmp_path).iloc[0]

    assert row["depth_frac"] == pytest.approx(0.5)
    assert bool(row["probe_acc_passed"])
    assert bool(row["perplexity_passed"])
    assert row["metric"] == "overall"
