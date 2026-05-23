"""Schema sanity checks on the released parquets and tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.parametrize("name,expected_cols", [
    ("logit", {"model_id", "family", "generation", "size", "variant",
               "num_params", "benchmark", "prompt_mode", "metric", "value"}),
    ("probe", {"model_id", "family", "generation", "size", "variant",
               "attribute", "layer", "layer_normalized", "mean_accuracy",
               "std_accuracy"}),
    ("intervention", {"model_id", "family", "generation", "size", "variant",
                       "benchmark", "prompt_mode", "attribute", "method",
                       "layer_idx", "depth_frac", "metric", "value"}),
])
def test_parquet_schema(name: str, expected_cols: set[str]):
    df = pd.read_parquet(DATA / "aggregated" / f"{name}.parquet")
    assert not df.empty, f"{name}.parquet is empty"
    missing = expected_cols - set(df.columns)
    assert not missing, f"{name}.parquet missing columns: {sorted(missing)}"


def test_logit_parquet_covers_four_benchmarks():
    df = pd.read_parquet(DATA / "aggregated" / "logit.parquet")
    benches = set(df["benchmark"].unique())
    # The four headline benchmarks must all be present; other benchmarks
    # (e.g. multilingual variants from older runs) are allowed.
    assert {"crows_pairs", "stereoset", "bbq", "iat"} <= benches


def test_intervention_parquet_has_eight_pairs():
    df = pd.read_parquet(DATA / "aggregated" / "intervention.parquet")
    # 8 probing-subset pairs × 2 variants = 16 distinct model_ids.
    assert df["model_id"].nunique() == 16


def test_intervention_parquet_has_sanity_columns():
    df = pd.read_parquet(DATA / "aggregated" / "intervention.parquet")
    # The sanity-gate columns are what probing_stats.py filters on.
    assert {"probe_acc_post", "probe_acc_passed",
            "perplexity_ratio", "perplexity_passed"} <= set(df.columns)


def test_probe_per_layer_csv():
    df = pd.read_csv(DATA / "tables" / "probe_per_layer.csv")
    assert {"model_id", "family", "variant", "attribute", "layer",
            "layer_normalized", "mean_accuracy"} <= set(df.columns)
    assert (df["mean_accuracy"].between(0.0, 1.0)).all()


def test_direction_cosines_in_range():
    df = pd.read_csv(DATA / "tables" / "direction_cosines.csv")
    # Cosines are bounded in [-1, 1] by definition.
    assert (df["cosine"].between(-1.001, 1.001)).all()


def test_random_pair_baseline_size():
    df = pd.read_csv(DATA / "tables" / "random_pair_cosine_baseline.csv")
    # The number 656 is the matched-hidden-dim comparison count cited
    # in §3.3; if this drifts, the §3.3 paper claim drifts with it.
    assert len(df) == 656
