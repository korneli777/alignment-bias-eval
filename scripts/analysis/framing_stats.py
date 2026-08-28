"""Recoverability-framing analysis for CrowS-Pairs and BBQ.

CrowS-Pairs rebound comes from `logit.parquet`; the BBQ decomposition is
recomputed per item from `bbq_items.parquet` with the same `bbq_metrics`
function the benchmark runner uses, so the released per-item table is what
backs the published means rather than a parallel implementation.

Usage:
    python scripts/analysis/framing_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest

from biaseval.benchmarks.bbq_metrics import bbq_metrics
from biaseval.benchmarks.framings import FRAMINGS

sys.path.insert(0, str(Path(__file__).parent))
from _common import render_section

from biaseval.analysis.statistics import holm_bonferroni

REPO = Path(__file__).resolve().parents[2]

def _pairs() -> list[tuple[str, str]]:
    cfg = yaml.safe_load((REPO / "configs/models.yaml").read_text())
    return [(m["base_id"], m["instruct_id"])
            for fd in cfg["families"].values()
            for gen in fd["generations"] for m in gen["models"]]


def crows_rebound(logit: pd.DataFrame) -> dict[str, dict]:
    """Per condition: mean rebound, pairs rebounding, sign-test p, gender rebound."""
    c = logit[logit.benchmark == "crows_pairs"]
    lookup = {(m, p, k): v for m, p, k, v in
              zip(c.model_id, c.prompt_mode, c.metric, c.value, strict=True)}
    out = {}
    for fid in FRAMINGS:
        reb, gen = [], []
        for _, inst in _pairs():
            f, i = lookup.get((inst, fid, "overall")), lookup.get((inst, "instruct", "overall"))
            if f is not None and i is not None:
                reb.append(f - i)
            fg, ig = lookup.get((inst, fid, "gender")), lookup.get((inst, "instruct", "gender"))
            if fg is not None and ig is not None:
                gen.append(fg - ig)
        n_pos = sum(1 for x in reb if x > 0)
        out[fid] = {
            "n": len(reb), "rebound": float(np.mean(reb)), "n_pos": n_pos,
            "p": binomtest(n_pos, len(reb), 0.5, alternative="greater").pvalue,
            "gender": float(np.mean(gen)),
        }
    adjusted = holm_bonferroni({fid: values["p"] for fid, values in out.items()})
    for fid, values in adjusted.items():
        out[fid]["p_holm"] = values["p_adj"]
    return out


def bbq_decomposition(items: pd.DataFrame) -> pd.DataFrame:
    """Per (model, condition) deferral rate, conditional bias, and aggregate."""
    rows = []
    for (model_id, mode), g in items.groupby(["model_id", "prompt_mode"], observed=True):
        m = bbq_metrics(g.to_dict("records"))
        rows.append({"model_id": model_id, "prompt_mode": mode,
                     "r_def": m["deferral_rate"], "s_cond": m["conditional_bias"],
                     "s_amb": m["bias_ambig"]})
    return pd.DataFrame(rows)


def main() -> int:
    """Compute and print the recoverability-framing summaries."""
    logit = pd.read_parquet(REPO / "data/aggregated/logit.parquet")
    items = pd.read_parquet(REPO / "data/aggregated/bbq_items.parquet")

    crows = crows_rebound(logit)
    crows_rows = []
    for fid, values in crows.items():
        name = FRAMINGS[fid].grounding.split(" (")[0]
        crows_rows.append((
            name,
            f"rebound={values['rebound']:+.3f} pp; "
            f"positive={values['n_pos']}/{values['n']}; "
            f"gender={values['gender']:+.3f} pp; Holm p={values['p_holm']:.4g}",
        ))

    per = bbq_decomposition(items)
    means = per.groupby("prompt_mode", observed=True)[["r_def", "s_cond", "s_amb"]].mean()
    order = ["raw", "instruct", *[mode for mode in FRAMINGS if mode in means.index]]
    bbq_rows = []
    for mode in order:
        label = {"raw": "Base, no template", "instruct": "Instruct, template"}.get(
            mode, FRAMINGS[mode].grounding.split(" (")[0] if mode in FRAMINGS else mode)
        values = means.loc[mode]
        n_models = int((per["prompt_mode"] == mode).sum())
        bbq_rows.append((
            label,
            f"r_def={values.r_def:.3f}; s_cond={values.s_cond:.3f}; "
            f"s_amb={values.s_amb:+.3f}; n={n_models}",
        ))

    inst = per[per.prompt_mode == "instruct"].set_index("model_id").r_def
    hi_inst = set(inst[inst > 0.90].index)
    inst2base = {i: b for b, i in _pairs()}
    drop = hi_inst | {inst2base[i] for i in hi_inst}
    sub = (per[~per.model_id.isin(drop)]
           .groupby("prompt_mode", observed=True)[["s_cond", "s_amb"]].mean())
    framing_modes = [mode for mode in FRAMINGS if mode in sub.index]
    subset_rows = [
        ("Instruct pairs excluded", f"{len(hi_inst)}/{len(inst)}"),
        ("Base", f"s_cond={sub.loc['raw', 's_cond']:.3f}; s_amb={sub.loc['raw', 's_amb']:+.3f}"),
        ("Instruct", f"s_cond={sub.loc['instruct', 's_cond']:.3f}; s_amb={sub.loc['instruct', 's_amb']:+.3f}"),
        (
            "Framing range",
            f"s_cond=[{min(sub.loc[f, 's_cond'] for f in framing_modes):.3f}, "
            f"{max(sub.loc[f, 's_cond'] for f in framing_modes):.3f}]; "
            f"s_amb=[{min(sub.loc[f, 's_amb'] for f in framing_modes):+.3f}, "
            f"{max(sub.loc[f, 's_amb'] for f in framing_modes):+.3f}]",
        ),
    ]

    render_section("CrowS-Pairs rebound per framing", crows_rows)
    render_section("BBQ decomposition per condition", bbq_rows)
    render_section("Sensitivity: instruct deferral ≤ 90%", subset_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
