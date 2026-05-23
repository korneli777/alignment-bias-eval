"""Probing accuracy, gender-direction cosines, and INLP/LEACE intervention.

Computes:
  - Per-pair peak probe accuracy (base vs instruct), mean and Spearman ρ
    across depth-binned curves
  - Cosine similarity between base and instruct gender directions per
    layer-pair, with a random-pair noise floor
  - INLP and LEACE intervention deltas on the gender CrowS category, and
    the mechanism-specificity ratio against non-gender categories
  - Sanity-gate counts (cells with post-projection probe accuracy ≤ chance
    + buffer and perplexity ratio ≤ 1.5)
  - Per-pair INLP/LEACE bias deltas across all probed pairs

When run as a script, results are compared against the paper's reported
values as a release-correctness check.

Usage:
    python scripts/analysis/probing_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from _common import check_close, check_count, check_pct, check_within, render_section

REPO = Path(__file__).resolve().parents[2]


def probe_peak_per_pair(probe_df: pd.DataFrame) -> pd.DataFrame:
    g = probe_df[probe_df["attribute"] == "gender"].copy()
    peak_idx = g.groupby(["model_id", "variant"])["mean_accuracy"].idxmax()
    peaks = g.loc[peak_idx, ["family", "generation", "size", "variant", "mean_accuracy"]]
    wide = peaks.pivot_table(
        index=["family", "generation", "size"],
        columns="variant", values="mean_accuracy", aggfunc="first",
    ).reset_index()
    return wide.dropna(subset=["base", "instruct"])


def depth_binned_means(probe_df: pd.DataFrame) -> pd.DataFrame:
    """Across the 8 probed pairs, mean probe accuracy per (depth_bin × variant)."""
    g = probe_df[probe_df["attribute"] == "gender"].copy()
    g["depth_bin"] = pd.cut(
        g["layer_normalized"], bins=np.linspace(0, 1.0001, 11),
        labels=[f"{i/10:.1f}" for i in range(10)], include_lowest=True,
    )
    return (g.groupby(["depth_bin", "variant"], observed=True)["mean_accuracy"]
              .mean().unstack("variant"))


def intervention_bias_delta(intv_df: pd.DataFrame, logit_df: pd.DataFrame,
                             sanity_filter: bool = True) -> pd.DataFrame:
    """Long-format dataframe of (intervention_value - baseline) per cell."""
    # Baseline = same (model, benchmark, prompt_mode, metric), no intervention.
    baseline = (logit_df[["model_id", "benchmark", "prompt_mode", "metric", "value"]]
                .rename(columns={"value": "baseline"}))
    iv = intv_df.merge(
        baseline, on=["model_id", "benchmark", "prompt_mode", "metric"], how="left",
    )
    iv["bias_delta"] = iv["baseline"] - iv["value"]
    if sanity_filter:
        iv = iv[iv["probe_acc_passed"].fillna(False)
                 & iv["perplexity_passed"].fillna(False)]
    return iv


def run() -> int:
    probe_df  = pd.read_parquet(REPO / "data/aggregated/probe.parquet")
    intv_df   = pd.read_parquet(REPO / "data/aggregated/intervention.parquet")
    logit_df  = pd.read_parquet(REPO / "data/aggregated/logit.parquet")
    cos_df    = pd.read_csv(REPO / "data/tables/direction_cosines.csv")
    baseline  = pd.read_csv(REPO / "data/tables/random_pair_cosine_baseline.csv")

    # ─── Peak probe accuracy across the 8 pairs ──────────────────────────
    peaks = probe_peak_per_pair(probe_df)
    mean_base    = float(peaks["base"].mean())
    mean_inst    = float(peaks["instruct"].mean())

    # ─── Spearman ρ between base and instruct curves (binned) ─────────────
    bins = depth_binned_means(probe_df)
    rho, _ = spearmanr(bins["base"], bins["instruct"])

    accuracy_checks = [
        check_close("Mean peak probe accuracy (base)",     0.81, mean_base, tol=0.01),
        check_close("Mean peak probe accuracy (instruct)", 0.78, mean_inst, tol=0.02),
        check_close("Spearman ρ base vs instruct curves",  0.91, float(rho), tol=0.02),
    ]

    # ─── Direction cosines (gender) ───────────────────────────────────────
    g_cos = cos_df[cos_df["attribute"] == "gender"]
    cos_median = float(g_cos["cosine"].median())
    cos_frac_08 = float((g_cos["cosine"] > 0.8).mean())
    cos_n = int(len(g_cos))
    cosine_checks = [
        check_count("Cosine n (pair × layer)", 280, cos_n),
        check_close("Cosine median (gender)",   0.86, cos_median,   tol=0.005),
        check_pct("Cosine fraction > 0.8",     0.75, cos_frac_08, tol=0.01),
    ]

    # ─── Random-pair baseline ─────────────────────────────────────────────
    baseline_n      = int(len(baseline))
    baseline_median = float(baseline["cosine"].median())
    baseline_max    = float(baseline["cosine"].abs().max())
    baseline_checks = [
        check_count("Random-pair baseline n", 656, baseline_n),
        check_close("Random-pair baseline median", 0.006, baseline_median, tol=0.005),
        check_close("Random-pair baseline |max| < 0.5",
                    0.05, baseline_max, tol=0.5),
    ]

    # ─── INLP / LEACE on the gender CrowS category ────────────────────────
    iv = intervention_bias_delta(intv_df, logit_df, sanity_filter=True)
    g_cat = iv[(iv["attribute"] == "gender") & (iv["benchmark"] == "crows_pairs")
                & (iv["metric"] == "gender")]
    inlp_mean  = float(g_cat[g_cat["method"] == "inlp"]["bias_delta"].mean())
    leace_mean = float(g_cat[g_cat["method"] == "leace"]["bias_delta"].mean())
    intervention_checks = [
        check_close("Cross-pair INLP gender Δ",  0.21, inlp_mean,  tol=0.05),
        check_close("Cross-pair LEACE gender Δ", 0.24, leace_mean, tol=0.05),
    ]

    # ─── Mechanism specificity (gender ablation, mean |Δ| non-gender) ─────
    other_cats = ["race-color", "religion", "age", "nationality",
                  "disability", "physical-appearance", "sexual-orientation",
                  "socioeconomic"]
    iv_cats = iv[(iv["attribute"] == "gender")
                  & (iv["benchmark"] == "crows_pairs")
                  & iv["metric"].isin(other_cats)]
    non_gender_abs = (iv_cats.groupby(["method", "metric"])["bias_delta"].mean()
                       .abs().groupby("method").mean())
    inlp_ratio  = inlp_mean  / float(non_gender_abs.loc["inlp"])
    leace_ratio = leace_mean / float(non_gender_abs.loc["leace"])
    specificity_checks = [
        check_close("Specificity ratio INLP",  1.5, inlp_ratio,  tol=0.3),
        check_close("Specificity ratio LEACE", 1.5, leace_ratio, tol=0.3),
    ]

    # ─── Sanity-gate exclusions ───────────────────────────────────────────
    raw_iv = intv_df[(intv_df["attribute"] == "gender")
                      & (intv_df["benchmark"] == "crows_pairs")]
    cells = (raw_iv[["model_id", "method", "layer_idx",
                     "probe_acc_passed", "perplexity_passed"]]
              .drop_duplicates())
    n_total = len(cells)
    failed = cells[~(cells["probe_acc_passed"].fillna(False)
                     & cells["perplexity_passed"].fillna(False))]
    n_failed = len(failed)
    n_failed_qwen = int(failed["model_id"].str.contains("Qwen2.5-7B", na=False).sum())
    exclusion_checks = [
        check_count("Total intervention cells", 160, n_total),
        check_count("Cells failing sanity gate", 12, n_failed),
        check_count("Qwen 2.5 7B share of failures", 11, n_failed_qwen),
    ]

    # ─── Per-pair INLP/LEACE breakdown (8 probed pairs) ───────────────────
    per_pair = (g_cat.groupby(["family", "generation", "size", "method"])["bias_delta"]
                  .mean().unstack("method"))

    def cell(family, generation, size, method) -> float:
        try:
            return float(per_pair.loc[(family, generation, size), method])
        except KeyError:
            return float("nan")

    pair_checks = [
        check_close("Llama 2 7B INLP",            +1.30, cell("llama",   "Llama 2",         "7B",  "inlp"),  tol=0.2),
        check_close("Llama 2 7B LEACE",           +2.14, cell("llama",   "Llama 2",         "7B",  "leace"), tol=0.2),
        check_close("Mistral Small 3 24B INLP",   +1.03, cell("mistral", "Mistral Small 3", "24B", "inlp"),  tol=0.2),
        check_close("Mistral v0.3 7B INLP",       -1.36, cell("mistral", "Mistral v0.3",    "7B",  "inlp"),  tol=0.2),
        check_close("Mistral v0.3 7B LEACE",      -1.13, cell("mistral", "Mistral v0.3",    "7B",  "leace"), tol=0.2),
        check_within("Gemma 2 9B INLP  within ±0.2",         -0.2, +0.2, cell("gemma",   "Gemma 2",         "9B",  "inlp")),
        check_within("Gemma 3 4B INLP  within ±0.2",         -0.2, +0.2, cell("gemma",   "Gemma 3",         "4B",  "inlp")),
        check_within("Llama 3.1 8B INLP within ±0.2",        -0.2, +0.2, cell("llama",   "Llama 3.1",       "8B",  "inlp")),
    ]

    fails = 0
    fails += render_section("§3.3 Probe peak accuracy + curves",    accuracy_checks)
    fails += render_section("§3.3 Direction cosines (gender)",      cosine_checks)
    fails += render_section("§3.3 Random-pair cosine baseline",     baseline_checks)
    fails += render_section("§3.3 INLP / LEACE on gender CrowS",    intervention_checks)
    fails += render_section("§3.3 Mechanism specificity",           specificity_checks)
    fails += render_section("§3.3 Sanity-gate exclusions",          exclusion_checks)
    fails += render_section("§3.3 Per-pair INLP/LEACE breakdown",   pair_checks)
    return fails


if __name__ == "__main__":
    sys.exit(0 if run() == 0 else 1)
