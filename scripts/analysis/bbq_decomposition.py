"""BBQ deferral / conditional-bias decomposition + CrowS jailbreak rebound.

Per-pair decomposition of the BBQ ambiguous-context score into deferral
rate and conditional bias, and the jailbreak-prompt rebound on CrowS-
Pairs (overall and per bias category).

When run as a script, results are compared against the paper's reported
values as a release-correctness check.

Usage:
    python scripts/analysis/bbq_decomposition.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from _common import CheckResult, check_close, check_count, check_pct, render_section

REPO = Path(__file__).resolve().parents[2]


def load_pairs() -> list[tuple[str, str, str, str, str]]:
    with open(REPO / "configs/models.yaml") as f:
        cfg = yaml.safe_load(f)
    return [
        (entry["base_id"], entry["instruct_id"], fam, gen["name"], entry["size"])
        for fam, family in cfg["families"].items()
        for gen in family["generations"]
        for entry in gen["models"]
    ]


def bbq_per_pair(logit_df: pd.DataFrame) -> pd.DataFrame:
    """Per (base, instruct) pair: deferral_base, deferral_inst, etc."""
    pairs = load_pairs()

    def score(model_id, prompt_mode, metric):
        sub = logit_df[(logit_df["model_id"] == model_id)
                        & (logit_df["benchmark"] == "bbq")
                        & (logit_df["prompt_mode"] == prompt_mode)
                        & (logit_df["metric"] == metric)]
        return float(sub["value"].iloc[0]) if not sub.empty else float("nan")

    rows = []
    for base_id, inst_id, family, generation, size in pairs:
        rows.append({
            "family": family, "generation": generation, "size": size,
            "deferral_base":  score(base_id, "raw",       "overall_deferral_rate"),
            "deferral_inst":  score(inst_id, "instruct",  "overall_deferral_rate"),
            "cond_base":      score(base_id, "raw",       "overall_conditional_bias"),
            "cond_inst":      score(inst_id, "instruct",  "overall_conditional_bias"),
        })
    return pd.DataFrame(rows)


def bbq_recoverability(logit_df: pd.DataFrame, *, n_iter: int = 10_000,
                       seed: int = 42) -> dict:
    """Per-pair Δ on overall_bias_ambig = (instruct+persona) − base, plus a
    paired bootstrap 95% CI. Only pairs with all three modes present (base
    raw, instruct chat-templated, instruct jailbreak) are included.
    """
    pairs = load_pairs()

    def score(model_id, prompt_mode):
        sub = logit_df[(logit_df["model_id"] == model_id)
                        & (logit_df["benchmark"] == "bbq")
                        & (logit_df["prompt_mode"] == prompt_mode)
                        & (logit_df["metric"] == "overall_bias_ambig")]
        return float(sub["value"].iloc[0]) if not sub.empty else float("nan")

    base, inst, jail = [], [], []
    for base_id, inst_id, family, generation, size in pairs:
        b = score(base_id, "raw")
        i = score(inst_id, "instruct")
        j = score(inst_id, "jailbreak")
        if any(np.isnan(v) for v in (b, i, j)):
            continue
        base.append(b); inst.append(i); jail.append(j)
    base = np.asarray(base); inst = np.asarray(inst); jail = np.asarray(jail)

    deltas = jail - base
    rng = np.random.default_rng(seed)
    n = len(deltas)
    boots = np.array([deltas[rng.integers(0, n, n)].mean() for _ in range(n_iter)])

    return {
        "n_pairs":       n,
        "base_mean":     float(base.mean()),
        "instruct_mean": float(inst.mean()),
        "jailbreak_mean": float(jail.mean()),
        "delta_point":   float(deltas.mean()),
        "delta_ci_lo":   float(np.quantile(boots, 0.025)),
        "delta_ci_hi":   float(np.quantile(boots, 0.975)),
    }


def crows_jailbreak_rebound(logit_df: pd.DataFrame) -> pd.DataFrame:
    """Per instruct model: overall stereo-win-rate under raw / instruct / jailbreak."""
    pairs = load_pairs()

    def score(model_id, prompt_mode, metric):
        sub = logit_df[(logit_df["model_id"] == model_id)
                        & (logit_df["benchmark"] == "crows_pairs")
                        & (logit_df["prompt_mode"] == prompt_mode)
                        & (logit_df["metric"] == metric)]
        return float(sub["value"].iloc[0]) if not sub.empty else float("nan")

    rows = []
    for base_id, inst_id, family, generation, size in pairs:
        rows.append({
            "family": family, "generation": generation, "size": size,
            "instruct_id":    inst_id,
            "base_raw":       score(base_id, "raw",       "overall"),
            "instruct_raw":   score(inst_id, "raw",       "overall"),
            "instruct_chat":  score(inst_id, "instruct",  "overall"),
            "instruct_jb":    score(inst_id, "jailbreak", "overall"),
        })
    df = pd.DataFrame(rows)
    df["rebound"] = df["instruct_jb"] - df["instruct_chat"]
    return df


def crows_per_category_rebound(logit_df: pd.DataFrame) -> pd.DataFrame:
    """Per bias category × pair: jailbreak − instruct stereotype score."""
    cats = [
        "gender", "race-color", "religion", "age", "nationality",
        "disability", "physical-appearance", "sexual-orientation", "socioeconomic",
    ]
    sub = logit_df[(logit_df["benchmark"] == "crows_pairs")
                    & (logit_df["variant"] == "instruct")
                    & logit_df["prompt_mode"].isin(["instruct", "jailbreak"])
                    & logit_df["metric"].isin(cats)]
    wide = sub.pivot_table(
        index=["family", "generation", "size", "metric"],
        columns="prompt_mode", values="value", aggfunc="first",
    ).reset_index().rename(columns={"metric": "category"})
    if "jailbreak" in wide.columns and "instruct" in wide.columns:
        wide["rebound"] = wide["jailbreak"] - wide["instruct"]
    return wide


def run() -> int:
    logit_df = pd.read_parquet(REPO / "data/aggregated/logit.parquet")

    # ─── Deferral + conditional bias ──────────────────────────────────────
    bbq = bbq_per_pair(logit_df)

    mean_def_base = float(bbq["deferral_base"].mean())
    mean_def_inst = float(bbq["deferral_inst"].mean())
    n_def_up = int(((bbq["deferral_inst"] - bbq["deferral_base"]) > 0).sum())

    qwen = bbq[(bbq["family"] == "qwen") & (bbq["generation"] == "Qwen 2.5")
                & (bbq["size"] == "7B")]
    qwen_def_base = float(qwen["deferral_base"].iloc[0]) if not qwen.empty else float("nan")
    qwen_def_inst = float(qwen["deferral_inst"].iloc[0]) if not qwen.empty else float("nan")

    deferral_checks = [
        check_pct("Mean deferral base",        0.28, mean_def_base, tol=0.02),
        check_pct("Mean deferral instruct",    0.53, mean_def_inst, tol=0.02),
        check_count("Pairs where deferral rises", 23, n_def_up, of=27),
        check_pct("Qwen 2.5 7B deferral (base)",     0.42, qwen_def_base, tol=0.02),
        check_pct("Qwen 2.5 7B deferral (instruct)", 0.97, qwen_def_inst, tol=0.02),
    ]

    # ─── Jailbreak rebound on CrowS ────────────────────────────────────────
    cp = crows_jailbreak_rebound(logit_df)
    n_lift = int((cp["rebound"] > 0).sum())
    mean_rebound = float(cp["rebound"].mean())
    rebound_checks = [
        check_count("CrowS pairs with jailbreak > instruct", 24, n_lift, of=27),
        check_close("Mean CrowS jailbreak rebound (pp)", 2.8, mean_rebound, tol=0.5),
    ]

    # ─── Per-category rebound (paper only cites gender) ────────────────────
    cat = crows_per_category_rebound(logit_df)
    gender_rebound = float(cat[cat["category"] == "gender"]["rebound"].mean())
    category_checks = [
        check_close("Mean rebound — gender category", 4.2, gender_rebound, tol=0.5),
    ]

    # ─── BBQ persona-injection recoverability ──────────────────────────────
    rec = bbq_recoverability(logit_df)
    ci_contains_zero = rec["delta_ci_lo"] <= 0.0 <= rec["delta_ci_hi"]
    recoverability_checks = [
        check_count("n pairs (matched)", 27, rec["n_pairs"]),
        check_close("Mean base bias_ambig",            0.033, rec["base_mean"],      tol=0.005),
        check_close("Mean instruct bias_ambig",        0.028, rec["instruct_mean"],  tol=0.005),
        check_close("Mean instruct+persona bias_ambig", 0.033, rec["jailbreak_mean"], tol=0.005),
        check_close("Paired Δ_recover (point)",        0.001, rec["delta_point"],    tol=0.005),
        CheckResult(
            label="Δ_recover 95% CI contains 0",
            paper="[≤0, ≥0]",
            computed=f"[{rec['delta_ci_lo']:+.3f}, {rec['delta_ci_hi']:+.3f}]",
            passed=ci_contains_zero,
        ),
    ]

    fails = 0
    fails += render_section("§3.2 BBQ deferral + conditional bias", deferral_checks)
    fails += render_section("§3.2 CrowS jailbreak rebound", rebound_checks)
    fails += render_section("§3.2 Per-category rebound", category_checks)
    fails += render_section("§3.2 BBQ persona-injection recoverability", recoverability_checks)
    return fails


if __name__ == "__main__":
    sys.exit(0 if run() == 0 else 1)
