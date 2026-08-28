"""Unit tests for the BBQ deferral decomposition."""

from __future__ import annotations

import math

import pytest

from biaseval.benchmarks.bbq_metrics import bbq_metrics


def test_bbq_metrics_decomposes_mixed_predictions():
    records = [
        {"is_unknown_pred": True, "is_biased_pred": False},
        {"is_unknown_pred": False, "is_biased_pred": True},
        {"is_unknown_pred": False, "is_biased_pred": True},
        {"is_unknown_pred": False, "is_biased_pred": False},
    ]

    result = bbq_metrics(records)

    assert result["deferral_rate"] == pytest.approx(0.25)
    assert result["conditional_bias"] == pytest.approx(2 / 3)
    assert result["bias_ambig"] == pytest.approx(0.25)
    assert result["n_committed"] == 3


def test_bbq_metrics_handles_all_deferrals():
    result = bbq_metrics(
        [
            {"is_unknown_pred": True, "is_biased_pred": False},
            {"is_unknown_pred": True, "is_biased_pred": False},
        ]
    )

    assert result["deferral_rate"] == 1.0
    assert math.isnan(result["conditional_bias"])
    assert result["bias_ambig"] == 0.0
    assert result["n_committed"] == 0


def test_bbq_metrics_handles_empty_input():
    result = bbq_metrics([])

    assert result["n"] == 0
    assert result["bias_ambig"] == 0.0
    assert math.isnan(result["conditional_bias"])
