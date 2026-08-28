"""BBQ alignment effect, decomposed into deferral and conditional bias (§3.2).

Per-pair decomposition of the BBQ ambiguous-context score into a deferral
rate and a conditional bias on committed answers, comparing base against
instruct. The aggregate score barely moves between the two because the
components rise together and cancel in the product, which is the point of
reporting them separately.

The recoverability framings are checked in `framing_stats.py`, against the
per-item table rather than the full-split summaries: the framings scored a
6,001-item subsample, so they are not comparable to these full-split numbers.

When run as a script, the decomposition and sensitivity analysis are printed
from the released aggregate.

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
from _common import render_section

REPO = Path(__file__).resolve().parents[2]


def load_pairs() -> list[tuple[str, str, str, str, str]]:
    """Flatten configs/models.yaml into (base_id, instruct_id, family, gen, size).

    Every analysis is paired, so the registry is read as pairs rather than as a
    flat list of checkpoints. This is the one place that layout is unpacked.
    """
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


def bbq_alignment_delta(logit_df: pd.DataFrame, *, n_iter: int = 10_000,
                        seed: int = 42) -> dict:
    """Paired Δ on overall_bias_ambig (instruct − base) with a bootstrap 95% CI."""
    def score(model_id, prompt_mode):
        sub = logit_df[(logit_df["model_id"] == model_id)
                        & (logit_df["benchmark"] == "bbq")
                        & (logit_df["prompt_mode"] == prompt_mode)
                        & (logit_df["metric"] == "overall_bias_ambig")]
        return float(sub["value"].iloc[0]) if not sub.empty else float("nan")

    base, inst = [], []
    for base_id, inst_id, *_ in load_pairs():
        b, i = score(base_id, "raw"), score(inst_id, "instruct")
        if not (np.isnan(b) or np.isnan(i)):
            base.append(b)
            inst.append(i)
    base, inst = np.asarray(base), np.asarray(inst)

    deltas = inst - base
    rng = np.random.default_rng(seed)
    n = len(deltas)
    boots = np.array([deltas[rng.integers(0, n, n)].mean() for _ in range(n_iter)])
    return {
        "n_pairs":     n,
        "base_mean":   float(base.mean()),
        "instruct_mean": float(inst.mean()),
        "delta_point": float(deltas.mean()),
        "delta_ci_lo": float(np.quantile(boots, 0.025)),
        "delta_ci_hi": float(np.quantile(boots, 0.975)),
    }


def main() -> int:
    """Compute and print the base-to-instruct BBQ decomposition."""
    logit_df = pd.read_parquet(REPO / "data/aggregated/logit.parquet")
    bbq = bbq_per_pair(logit_df)

    d_def = bbq["deferral_inst"] - bbq["deferral_base"]
    d_cond = bbq["cond_inst"] - bbq["cond_base"]
    n_pairs = len(bbq)
    decomposition_rows = [
        ("Mean deferral, base", f"{bbq['deferral_base'].mean():.1%}"),
        ("Mean deferral, instruct", f"{bbq['deferral_inst'].mean():.1%}"),
        ("Pairs where deferral rises", f"{int((d_def > 0).sum())}/{n_pairs}"),
        ("Mean conditional bias, base", f"{bbq['cond_base'].mean():.3f}"),
        ("Mean conditional bias, instruct", f"{bbq['cond_inst'].mean():.3f}"),
        ("Pairs where conditional bias rises", f"{int((d_cond > 0).sum())}/{n_pairs}"),
    ]

    delta = bbq_alignment_delta(logit_df)
    aggregate_rows = [
        ("Matched pairs", str(delta["n_pairs"])),
        ("Mean base s_amb", f"{delta['base_mean']:+.3f}"),
        ("Mean instruct s_amb", f"{delta['instruct_mean']:+.3f}"),
        (
            "Paired Δ s_amb (instruct − base)",
            f"{delta['delta_point']:+.3f}  95% CI "
            f"[{delta['delta_ci_lo']:+.3f}, {delta['delta_ci_hi']:+.3f}]",
        ),
    ]

    keep = bbq[bbq["deferral_inst"] <= 0.90]
    keep_d = keep["cond_inst"] - keep["cond_base"]
    subset_rows = [
        ("Pairs retained", f"{len(keep)}/{n_pairs}"),
        ("Mean conditional-bias change", f"{keep_d.mean():+.3f}"),
        ("Pairs with an increase", f"{int((keep_d > 0).sum())}/{len(keep)}"),
    ]

    render_section("BBQ deferral and conditional bias", decomposition_rows)
    render_section("BBQ aggregate score", aggregate_rows)
    render_section("Sensitivity: instruct deferral ≤ 90%", subset_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
