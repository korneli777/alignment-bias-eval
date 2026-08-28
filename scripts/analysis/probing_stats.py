"""Probing accuracy, gender-direction cosines, and concept erasure.

Computes paired peak probe accuracy, depth-binned curve agreement,
base–instruct direction cosines, random-pair baselines, INLP/LEACE effects,
mechanism specificity, and intervention sanity-gate counts.

Usage:
    python scripts/analysis/probing_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr, ttest_rel

sys.path.insert(0, str(Path(__file__).parent))
from _common import render_section

REPO = Path(__file__).resolve().parents[2]


def probe_peak_per_pair(probe_df: pd.DataFrame) -> pd.DataFrame:
    """Peak gender-probe accuracy per pair, with one column per variant."""
    gender = probe_df[probe_df["attribute"] == "gender"].copy()
    peak_idx = gender.groupby(["model_id", "variant"])["mean_accuracy"].idxmax()
    peaks = gender.loc[
        peak_idx, ["family", "generation", "size", "variant", "mean_accuracy"]
    ]
    wide = peaks.pivot_table(
        index=["family", "generation", "size"],
        columns="variant",
        values="mean_accuracy",
        aggfunc="first",
    ).reset_index()
    return wide.dropna(subset=["base", "instruct"])


def depth_binned_means(probe_df: pd.DataFrame) -> pd.DataFrame:
    """Mean gender-probe accuracy by normalized depth and variant."""
    gender = probe_df[probe_df["attribute"] == "gender"].copy()
    gender["depth_bin"] = pd.cut(
        gender["layer_normalized"],
        bins=np.linspace(0, 1.0001, 11),
        labels=[f"{i / 10:.1f}" for i in range(10)],
        include_lowest=True,
    )
    return (
        gender.groupby(["depth_bin", "variant"], observed=True)["mean_accuracy"]
        .mean()
        .unstack("variant")
    )


def intervention_bias_delta(
    intv_df: pd.DataFrame,
    logit_df: pd.DataFrame,
    sanity_filter: bool = True,
) -> pd.DataFrame:
    """Return baseline-minus-intervention bias for each intervention cell."""
    baseline = logit_df[
        ["model_id", "benchmark", "prompt_mode", "metric", "value"]
    ].rename(columns={"value": "baseline"})
    interventions = intv_df.merge(
        baseline,
        on=["model_id", "benchmark", "prompt_mode", "metric"],
        how="left",
    )
    interventions["bias_delta"] = interventions["baseline"] - interventions["value"]
    if sanity_filter:
        interventions = interventions[
            interventions["probe_acc_passed"].fillna(False)
            & interventions["perplexity_passed"].fillna(False)
        ]
    return interventions


def main() -> int:
    """Compute and print the probing and intervention summaries."""
    probe_df = pd.read_parquet(REPO / "data/aggregated/probe.parquet")
    intv_df = pd.read_parquet(REPO / "data/aggregated/intervention.parquet")
    logit_df = pd.read_parquet(REPO / "data/aggregated/logit.parquet")
    cos_df = pd.read_csv(REPO / "data/tables/direction_cosines.csv")
    baseline = pd.read_csv(REPO / "data/tables/random_pair_cosine_baseline.csv")

    peaks = probe_peak_per_pair(probe_df)
    mean_base = float(peaks["base"].mean())
    mean_instruct = float(peaks["instruct"].mean())
    peak_delta = peaks["instruct"] - peaks["base"]
    t_p = float(ttest_rel(peaks["instruct"], peaks["base"]).pvalue)
    paired_d = float(peak_delta.mean() / peak_delta.std(ddof=1))
    nonzero = peak_delta[peak_delta != 0]
    n_lower = int((nonzero < 0).sum())
    n_higher = int((nonzero > 0).sum())
    n_ties = int((peak_delta == 0).sum())
    sign_p = float(binomtest(n_lower, len(nonzero), 0.5, alternative="greater").pvalue)
    bins = depth_binned_means(probe_df)
    rho, _ = spearmanr(bins["base"], bins["instruct"])
    accuracy_rows = [
        ("Mean peak accuracy, base", f"{mean_base:.3f}"),
        ("Mean peak accuracy, instruct", f"{mean_instruct:.3f}"),
        ("Instruct lower / tied / higher", f"{n_lower} / {n_ties} / {n_higher}"),
        ("Paired t-test", f"p={t_p:.4g}"),
        ("Paired Cohen's d", f"{paired_d:+.3f}"),
        ("Strict sign test", f"p={sign_p:.4g}"),
        ("Spearman ρ, depth-binned curves", f"{float(rho):.3f}"),
    ]

    gender_cosines = cos_df[cos_df["attribute"] == "gender"]
    cosine_rows = [
        ("Pair-layer observations", str(len(gender_cosines))),
        ("Median cosine", f"{gender_cosines['cosine'].median():.3f}"),
        ("Fraction above 0.8", f"{(gender_cosines['cosine'] > 0.8).mean():.1%}"),
    ]
    baseline_rows = [
        ("Comparisons", str(len(baseline))),
        ("Median cosine", f"{baseline['cosine'].median():.3f}"),
        ("Maximum |cosine|", f"{baseline['cosine'].abs().max():.3f}"),
    ]

    interventions = intervention_bias_delta(intv_df, logit_df, sanity_filter=True)
    gender_crows = interventions[
        (interventions["attribute"] == "gender")
        & (interventions["benchmark"] == "crows_pairs")
        & (interventions["metric"] == "gender")
    ]
    inlp_mean = float(gender_crows[gender_crows["method"] == "inlp"]["bias_delta"].mean())
    leace_mean = float(gender_crows[gender_crows["method"] == "leace"]["bias_delta"].mean())
    intervention_rows = [
        ("INLP mean gender Δ", f"{inlp_mean:+.3f}"),
        ("LEACE mean gender Δ", f"{leace_mean:+.3f}"),
    ]

    other_categories = [
        "race-color",
        "religion",
        "age",
        "nationality",
        "disability",
        "physical-appearance",
        "sexual-orientation",
        "socioeconomic",
    ]
    non_gender = interventions[
        (interventions["attribute"] == "gender")
        & (interventions["benchmark"] == "crows_pairs")
        & interventions["metric"].isin(other_categories)
    ]
    non_gender_abs = (
        non_gender.groupby(["method", "metric"])["bias_delta"]
        .mean()
        .abs()
        .groupby("method")
        .mean()
    )
    specificity_rows = [
        ("INLP gender/non-gender ratio", f"{inlp_mean / non_gender_abs.loc['inlp']:.3f}"),
        ("LEACE gender/non-gender ratio", f"{leace_mean / non_gender_abs.loc['leace']:.3f}"),
    ]

    raw_interventions = intv_df[
        (intv_df["attribute"] == "gender") & (intv_df["benchmark"] == "crows_pairs")
    ]
    cells = raw_interventions[
        ["model_id", "method", "layer_idx", "probe_acc_passed", "perplexity_passed"]
    ].drop_duplicates()
    failed = cells[
        ~(
            cells["probe_acc_passed"].fillna(False)
            & cells["perplexity_passed"].fillna(False)
        )
    ]
    exclusion_rows = [
        ("Intervention cells", str(len(cells))),
        ("Cells failing a sanity gate", str(len(failed))),
        (
            "Failures from Qwen 2.5 7B",
            str(int(failed["model_id"].str.contains("Qwen2.5-7B", na=False).sum())),
        ),
    ]

    per_pair = (
        gender_crows.groupby(["family", "generation", "size", "method"])["bias_delta"]
        .mean()
        .unstack("method")
    )
    pair_rows = []
    for (family, generation, size), values in per_pair.iterrows():
        parts = [
            f"{method.upper()}={float(value):+.3f}"
            for method, value in values.items()
            if pd.notna(value)
        ]
        pair_rows.append((f"{family} / {generation} / {size}", "; ".join(parts)))

    render_section("Probe peak accuracy and curves", accuracy_rows)
    render_section("Base–instruct gender-direction cosines", cosine_rows)
    render_section("Random-pair cosine baseline", baseline_rows)
    render_section("INLP and LEACE on gender CrowS-Pairs", intervention_rows)
    render_section("Mechanism specificity", specificity_rows)
    render_section("Sanity-gate exclusions", exclusion_rows)
    render_section("Per-pair intervention effects", pair_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
