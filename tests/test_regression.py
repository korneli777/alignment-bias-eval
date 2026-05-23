"""Sanity checks on biaseval.analysis.regression.

These don't try to replicate the full paper-number gate (that lives in
test_paper_numbers.py); they verify the regression module runs end-to-end
on the cached parquet and returns finite, expected-sign coefficients.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biaseval.analysis.regression import (
    HEADLINE_METRIC,
    build_regression_frame,
    fit_summary_model,
)

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def reg_df() -> pd.DataFrame:
    logit_df = pd.read_parquet(DATA / "aggregated" / "logit.parquet")
    return build_regression_frame(logit_df)


def test_build_regression_frame_has_headline_metrics_only(reg_df):
    """build_regression_frame keeps only the per-benchmark headline metric."""
    for bench, metric in HEADLINE_METRIC.items():
        sub = reg_df[reg_df["benchmark"] == bench]
        assert not sub.empty
        assert set(sub["metric"].unique()) == {metric}


def test_fit_summary_model_runs_per_benchmark(reg_df):
    """Each per-benchmark OLS fit returns coefficients with finite p-values."""
    for bench in HEADLINE_METRIC:
        fit = fit_summary_model(reg_df, bench)
        assert "params" in fit, f"fit for {bench} returned no params: {fit}"
        # variant coefficient must exist and have a finite p-value.
        variant_term = next(
            (t for t in fit["params"]
             if "variant" in t and "instruct" in t),
            None,
        )
        assert variant_term is not None, f"no variant term in {bench} fit"
        assert np.isfinite(fit["pvalues"][variant_term])


def test_crows_variant_effect_negative(reg_df):
    """On CrowS-Pairs, the variant=instruct coefficient should be negative
    (alignment reduces stereotype score). This is what the paper reports."""
    fit = fit_summary_model(reg_df, "crows_pairs")
    variant_term = next(t for t in fit["params"]
                        if "variant" in t and "instruct" in t)
    assert fit["params"][variant_term] < 0
