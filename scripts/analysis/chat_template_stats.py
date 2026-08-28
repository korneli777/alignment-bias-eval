"""Chat-template-conditional bias analysis.

Computes the chat-template contribution to the alignment effect on each
benchmark via paired bootstrap over base/instruct pairs, plus per-pair
Cohen's d, cross-benchmark agreement counts, and sign tests.

Also emits tables/regression.tex — pooled OLS per benchmark with HC3
SEs, Holm correction across the four benchmark variant coefficients,
and the paired-bootstrap Δ_chat with 95% CI.

When run as a script, the computed estimates are printed and the regression
table fragment is regenerated from the released aggregates.

Usage:
    python scripts/analysis/chat_template_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yaml
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent))
from _common import render_section

REPO = Path(__file__).resolve().parents[2]

HEADLINE_METRIC = {
    "crows_pairs": "overall",
    "stereoset":   "overall_SS",
    "bbq":         "overall_bias_ambig",
    "iat":         "overall_abs_d",
}
# Transform into a "lower-is-less-biased" scale per benchmark so the
# sign of every Δ is comparable.
LOWER_IS_LESS = {
    "crows_pairs": lambda v: v,
    "stereoset":   lambda v: abs(v - 50),
    "bbq":         lambda v: abs(v),
    "iat":         lambda v: v,
}


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


def per_pair_deltas(logit_df: pd.DataFrame) -> pd.DataFrame:
    """For each (pair, benchmark), compute raw_delta and native_delta on the
    lower-is-less-biased scale."""
    pairs = load_pairs()

    def score(model_id: str, benchmark: str, prompt_mode: str) -> float:
        metric = HEADLINE_METRIC[benchmark]
        sub = logit_df[(logit_df["model_id"] == model_id)
                        & (logit_df["benchmark"] == benchmark)
                        & (logit_df["prompt_mode"] == prompt_mode)
                        & (logit_df["metric"] == metric)]
        return float(sub["value"].iloc[0]) if not sub.empty else float("nan")

    rows = []
    for base_id, inst_id, family, generation, size in pairs:
        for b in HEADLINE_METRIC:
            t = LOWER_IS_LESS[b]
            b_raw  = score(base_id, b, "raw")
            i_raw  = score(inst_id, b, "raw")
            i_chat = score(inst_id, b, "instruct")
            if any(np.isnan(x) for x in (b_raw, i_raw, i_chat)):
                continue
            rows.append({
                "family": family, "generation": generation, "size": size,
                "base_id": base_id, "instruct_id": inst_id, "benchmark": b,
                "raw_delta":    t(i_raw)  - t(b_raw),
                "native_delta": t(i_chat) - t(b_raw),
            })
    return pd.DataFrame(rows)


def delta_chat_bootstrap(deltas: pd.DataFrame, n_resamples: int = 10_000,
                         seed: int = 42) -> dict[str, dict]:
    """Paired bootstrap CI on Δ_chat = native_delta - raw_delta per benchmark."""
    out: dict[str, dict] = {}
    rng = np.random.default_rng(seed)
    for b in HEADLINE_METRIC:
        sub = deltas[deltas["benchmark"] == b]
        chat = (sub["native_delta"] - sub["raw_delta"]).to_numpy()
        n = len(chat)
        if n == 0:
            continue
        boot = chat[rng.integers(0, n, size=(n_resamples, n))].mean(axis=1)
        out[b] = {
            "theta": float(chat.mean()),
            "ci_lo": float(np.quantile(boot, 0.025)),
            "ci_hi": float(np.quantile(boot, 0.975)),
            "n": n,
        }
    return out


def load_crows_effect_sizes(path: Path) -> pd.DataFrame:
    """Read and validate the exact per-pair CrowS-Pairs effect-size table."""
    frame = pd.read_csv(path)
    required = {
        "base_id", "instruct_id", "condition", "n_items", "cohens_d",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"effect-size table is missing columns: {sorted(missing)}")
    if frame.duplicated(["base_id", "instruct_id", "condition"]).any():
        raise ValueError("effect-size table contains duplicate pair-condition rows")
    conditions = set(frame["condition"])
    if conditions != {"without_template", "with_template"}:
        raise ValueError(f"unexpected effect-size conditions: {sorted(conditions)}")
    coverage = frame.groupby(["base_id", "instruct_id"])["condition"].nunique()
    if not (coverage == 2).all():
        raise ValueError("each pair must have both scoring conditions")
    if not (frame["n_items"] > 0).all():
        raise ValueError("effect-size rows must contain at least one item")
    return frame


def all_agree_count(deltas: pd.DataFrame, condition: str) -> int:
    """Count pairs where all 4 benchmark deltas are negative under `condition`.

    condition ∈ {raw_delta, native_delta}
    """
    pivot = deltas.pivot_table(
        index=["family", "generation", "size"], columns="benchmark",
        values=condition, aggfunc="first",
    )
    benches = list(HEADLINE_METRIC.keys())
    has_all = pivot[benches].notna().all(axis=1)
    all_neg = (pivot[benches] < 0).all(axis=1)
    return int((has_all & all_neg).sum())


def sign_test(deltas: pd.DataFrame, benchmark: str, condition: str) -> dict[str, float]:
    """One-sided exact sign test for a reduction in benchmark-defined bias."""
    values = deltas.loc[deltas["benchmark"] == benchmark, condition].dropna()
    nonzero = values[values != 0]
    n_negative = int((nonzero < 0).sum())
    p = binomtest(
        n_negative, len(nonzero), 0.5, alternative="greater"
    ).pvalue
    return {"n_negative": n_negative, "n": len(nonzero), "p": float(p)}


def fit_pooled_ols(logit_df: pd.DataFrame, benchmark: str, condition: str):
    """Pooled OLS on one benchmark under one scoring condition.

    condition ∈ {without_template, with_template}
    Returns (beta_variant, hc3_se, pvalue, n).
    """
    pairs = load_pairs()
    pair_ids = {b for b, *_ in pairs} | {i for _, i, *_ in pairs}

    metric = HEADLINE_METRIC[benchmark]
    sub = logit_df[(logit_df["benchmark"] == benchmark)
                    & (logit_df["metric"] == metric)
                    & logit_df["model_id"].isin(pair_ids)].copy()
    if condition == "without_template":
        sub = sub[sub["prompt_mode"] == "raw"]
    else:
        # base uses raw, instruct uses chat-template scoring
        sub = sub[((sub["variant"] == "base")     & (sub["prompt_mode"] == "raw"))
                  | ((sub["variant"] == "instruct") & (sub["prompt_mode"] == "instruct"))]
    sub["log_params"] = np.log10(sub["num_params"].astype(float))

    formula = (
        "value ~ C(variant, Treatment('base'))"
        " + log_params + C(family) + C(generation)"
    )
    fit = smf.ols(formula, data=sub).fit(cov_type="HC3")
    term = next(t for t in fit.params.index if "variant" in t and "instruct" in t)
    return float(fit.params[term]), float(fit.bse[term]), float(fit.pvalues[term]), int(fit.nobs)


def holm_bonferroni(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down correction over a family of tests.

    Adjusted p-values are forced to be non-decreasing by carrying a running
    maximum down the sorted list. Without that, a later test could report a
    smaller adjusted p-value than an earlier, more significant one.
    """
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, p * (m - i))
        running = max(running, adj)
        out[k] = {"p": p, "p_adj": running, "reject": running < alpha}
    return out


def emit_regression_tex(rows: list[dict], out_path: Path) -> None:
    """Write the per-benchmark regression table fragment used in Table 3."""
    def stars(p: float) -> str:
        if p < 0.001:
            return "^{***}"
        if p < 0.01:
            return "^{**}"
        if p < 0.05:
            return "^{*}"
        return ""

    label_map = {
        "crows_pairs": "CrowS-Pairs",
        "stereoset":   "StereoSet",
        "bbq":         "BBQ",
        "iat":         "IAT",
    }
    lines = []
    for r in rows:
        b   = r["benchmark"]
        w0  = r["without_beta"]
        w1  = r["with_beta"]
        p0  = r["without_p_adj"]
        p1  = r["with_p_adj"]
        d   = r["delta_chat"]
        lo  = r["ci_lo"]
        hi  = r["ci_hi"]
        s0  = stars(p0)
        s1  = stars(p1)
        lines.append(
            f"{label_map[b]:12s} & ${w0:+.2f}{s0}$ & ${w1:+.2f}{s1}$ "
            f"& $\\mathbf{{{d:+.2f}}}$ & $[{lo:+.2f},\\;{hi:+.2f}]$ \\\\"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    """Run the chat-template analyses and print the computed estimates."""
    logit_df = pd.read_parquet(REPO / "data/aggregated/logit.parquet")
    deltas   = per_pair_deltas(logit_df)

    boots = delta_chat_bootstrap(deltas)
    delta_chat_rows = [
        (
            benchmark,
            f"{values['theta']:+.3f}  95% CI "
            f"[{values['ci_lo']:+.3f}, {values['ci_hi']:+.3f}]  n={values['n']}",
        )
        for benchmark, values in boots.items()
    ]

    effect_size_path = REPO / "data" / "tables" / "crows_pair_effect_sizes.csv"
    if effect_size_path.is_file():
        d_df = load_crows_effect_sizes(effect_size_path)
        max_d = float(d_df["cohens_d"].abs().max())
        effect_size_rows = [
            ("Pair-condition rows", str(len(d_df))),
            ("Maximum per-pair |d|", f"{max_d:.3f}"),
        ]
    else:
        effect_size_rows = []

    n_raw    = all_agree_count(deltas, "raw_delta")
    n_native = all_agree_count(deltas, "native_delta")
    n_pairs = len(load_pairs())
    agreement_rows = [
        ("Without template", f"{n_raw}/{n_pairs}"),
        ("With template", f"{n_native}/{n_pairs}"),
    ]

    signs = {
        (benchmark, condition): sign_test(deltas, benchmark, condition)
        for benchmark in ("crows_pairs", "stereoset", "iat")
        for condition in ("raw_delta", "native_delta")
    }
    sign_rows = []
    condition_label = {"raw_delta": "without template", "native_delta": "with template"}
    for (benchmark, condition), values in signs.items():
        sign_rows.append((
            f"{benchmark}, {condition_label[condition]}",
            f"{values['n_negative']}/{values['n']} reductions; p={values['p']:.4g}",
        ))

    rows = []
    pvals_without, pvals_with = {}, {}
    for b in HEADLINE_METRIC:
        b0, _, p0, _ = fit_pooled_ols(logit_df, b, "without_template")
        b1, _, p1, _ = fit_pooled_ols(logit_df, b, "with_template")
        pvals_without[b] = p0
        pvals_with[b]    = p1
        rows.append({
            "benchmark": b,
            "without_beta": b0, "without_p": p0,
            "with_beta":    b1, "with_p":    p1,
            "delta_chat":   boots[b]["theta"],
            "ci_lo":        boots[b]["ci_lo"],
            "ci_hi":        boots[b]["ci_hi"],
        })
    h_without = holm_bonferroni(pvals_without)
    h_with    = holm_bonferroni(pvals_with)
    for r in rows:
        r["without_p_adj"] = h_without[r["benchmark"]]["p_adj"]
        r["with_p_adj"]    = h_with[r["benchmark"]]["p_adj"]

    emit_regression_tex(rows, REPO / "tables/regression.tex")
    regression_rows = []
    for row in rows:
        regression_rows.extend([
            (
                f"{row['benchmark']}, without template",
                f"β={row['without_beta']:+.3f}; Holm p={row['without_p_adj']:.4g}",
            ),
            (
                f"{row['benchmark']}, with template",
                f"β={row['with_beta']:+.3f}; Holm p={row['with_p_adj']:.4g}",
            ),
        ])

    render_section("Chat-template contribution (paired bootstrap)", delta_chat_rows)
    if effect_size_rows:
        render_section("CrowS-Pairs per-pair effect sizes", effect_size_rows)
    else:
        print(
            "\n[CrowS-Pairs per-pair effect sizes]\n"
            "  Not available: build data/tables/crows_pair_effect_sizes.csv "
            "from the raw CrowS-Pairs results."
        )
    render_section("Cross-benchmark agreement", agreement_rows)
    render_section("Exact sign tests", sign_rows)
    render_section("Pooled OLS variant coefficients", regression_rows)
    print(f"\n  Wrote tables/regression.tex ({len(rows)} rows).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
