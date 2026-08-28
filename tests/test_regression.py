"""Unit tests for regression-frame construction and OLS fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from biaseval.analysis.regression import (
    HEADLINE_METRIC,
    build_regression_frame,
    fit_summary_model,
)


def _synthetic_logit() -> pd.DataFrame:
    rows = []
    for benchmark, metric in HEADLINE_METRIC.items():
        for i in range(24):
            variant = "instruct" if i % 2 else "base"
            rows.append(
                {
                    "model_id": f"model-{i}",
                    "family": f"family-{(i // 2) % 2}",
                    "generation": f"generation-{(i // 4) % 2}",
                    "variant": variant,
                    "num_params": 1_000_000_000 + i * 50_000_000,
                    "benchmark": benchmark,
                    "prompt_mode": "raw",
                    "metric": metric,
                    "value": 50.0 - 2.0 * (variant == "instruct") + 0.03 * i,
                }
            )
    rows.append({**rows[0], "metric": "unused_metric", "value": 999.0})
    return pd.DataFrame(rows)


def test_build_regression_frame_selects_headline_metrics():
    frame = build_regression_frame(_synthetic_logit())

    assert len(frame) == 24 * len(HEADLINE_METRIC)
    assert "unused_metric" not in set(frame["metric"])
    assert np.isfinite(frame["log_params"]).all()
    for benchmark, metric in HEADLINE_METRIC.items():
        assert set(frame.loc[frame["benchmark"] == benchmark, "metric"]) == {metric}


def test_fit_summary_model_recovers_synthetic_variant_effect():
    frame = build_regression_frame(_synthetic_logit())

    fit = fit_summary_model(frame, "crows_pairs")
    variant_term = next(term for term in fit["params"] if "variant" in term and "instruct" in term)

    assert fit["cov_type"] == "HC3"
    assert fit["n"] == 24
    assert fit["params"][variant_term] < -1.5
    assert np.isfinite(fit["pvalues"][variant_term])


def test_fit_summary_model_reports_missing_benchmark():
    frame = build_regression_frame(_synthetic_logit())

    fit = fit_summary_model(frame, "not-a-benchmark")

    assert fit["n"] == 0
    assert fit["model_type"] == "none"
