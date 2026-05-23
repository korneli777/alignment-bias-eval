"""BBQ deferral-vs-conditional-bias quadrant scatter (Δ instruct − base on each
axis). Saves figures/bbq_quadrant.{png,pdf}. (Paper Figure 4.)
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)

FIGS = Path("figures")
FIGS.mkdir(parents=True, exist_ok=True)

FAMILY_ORDER = ["llama", "qwen", "mistral", "gemma"]
FAMILY_DISPLAY = {"llama": "Llama", "qwen": "Qwen",
                  "mistral": "Mistral", "gemma": "Gemma"}
FAMILY_COLOURS = {
    "llama":   "#1f77b4",
    "qwen":    "#ff7f0e",
    "mistral": "#d62728",
    "gemma":   "#2ca02c",
}


def load() -> pd.DataFrame:
    df = pd.read_parquet("data/aggregated/logit.parquet")
    metrics = ["overall_deferral_rate", "overall_conditional_bias"]
    sub = df[(df["benchmark"] == "bbq") & df["metric"].isin(metrics)].copy()
    base = sub[(sub["variant"] == "base")     & (sub["prompt_mode"] == "raw")]
    inst = sub[(sub["variant"] == "instruct") & (sub["prompt_mode"] == "instruct")]
    bp = base.pivot_table(
        index=["family", "generation", "size"],
        columns="metric", values="value", aggfunc="first",
    ).rename(columns={
        "overall_deferral_rate":    "deferral_base",
        "overall_conditional_bias": "cond_base",
    })
    ip = inst.pivot_table(
        index=["family", "generation", "size"],
        columns="metric", values="value", aggfunc="first",
    ).rename(columns={
        "overall_deferral_rate":    "deferral_inst",
        "overall_conditional_bias": "cond_inst",
    })
    j = bp.join(ip).dropna().reset_index()
    j["delta_def"]  = j["deferral_inst"] - j["deferral_base"]
    j["delta_cond"] = j["cond_inst"]     - j["cond_base"]
    return j


def make_figure(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(3.4, 3.2))

    x_lim = max(np.abs(df["delta_def"].values).max(),  0.30) * 1.12
    y_lim = max(np.abs(df["delta_cond"].values).max(), 0.30) * 1.12
    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(-y_lim, y_lim)

    grid = "#eeeeee"
    for gx in (-0.4, 0.4, 0.8):
        ax.axvline(gx, color=grid, linewidth=0.4, zorder=0)
    for gy in (-0.2, 0.2, 0.4):
        ax.axhline(gy, color=grid, linewidth=0.4, zorder=0)

    ax.axhline(0, color="#999999", linestyle="--", linewidth=0.9, zorder=1)
    ax.axvline(0, color="#999999", linestyle="--", linewidth=0.9, zorder=1)

    for fam in FAMILY_ORDER:
        f = df[df["family"] == fam]
        if f.empty:
            continue
        ax.scatter(f["delta_def"], f["delta_cond"],
                   s=34, marker="o",
                   facecolor=FAMILY_COLOURS[fam],
                   edgecolor="white", linewidth=0.6, zorder=4)

    # Two outliers worth calling out: Mistral v0.1 7B (deferral drops
    # but conditional bias rises) and Qwen 3 8B (the one pair going the
    # debiasing-predicted direction on both axes).
    arrow_style = dict(arrowstyle="-", color="#888", lw=0.6,
                       shrinkA=0, shrinkB=4)
    annot_style = dict(fontsize=6.5, color="#444", zorder=6)

    mv01 = df[(df.family == "mistral") & (df.generation == "Mistral v0.1")
              & (df["size"] == "7B")]
    if not mv01.empty:
        r = mv01.iloc[0]
        ax.annotate(
            "Mistral v0.1, 7B",
            xy=(r["delta_def"], r["delta_cond"]),
            xytext=(r["delta_def"] + 0.02, r["delta_cond"] + 0.16),
            ha="left", va="bottom",
            arrowprops={**arrow_style, "connectionstyle": "arc3,rad=-0.20"},
            **annot_style,
        )

    q3_8b = df[(df.family == "qwen") & (df.generation == "Qwen 3")
               & (df["size"] == "8B")]
    if not q3_8b.empty:
        r = q3_8b.iloc[0]
        ax.annotate(
            "Qwen 3, 8B",
            xy=(r["delta_def"], r["delta_cond"]),
            xytext=(r["delta_def"] + 0.02, r["delta_cond"] - 0.12),
            ha="left", va="top",
            arrowprops={**arrow_style, "connectionstyle": "arc3,rad=0.20"},
            **annot_style,
        )

    label_kw = dict(fontsize=6.5, color="#666", style="italic",
                    transform=ax.transAxes, zorder=2)
    ax.text(0.98, 0.97, "more deferral\nmore cond. bias",
            ha="right", va="top",    **label_kw)
    ax.text(0.02, 0.97, "less deferral\nmore cond. bias",
            ha="left",  va="top",    **label_kw)
    ax.text(0.98, 0.03, "more deferral\nless cond. bias",
            ha="right", va="bottom", **label_kw)
    ax.text(0.02, 0.03, "less deferral\nless cond. bias",
            ha="left",  va="bottom", **label_kw)

    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(axis="both", labelsize=7, labelcolor="#555555", length=0)
    ax.set_xticks([-0.4, 0, 0.4, 0.8])
    ax.set_yticks([-0.2, 0, 0.2, 0.4])

    fig.subplots_adjust(left=0.16, right=0.84, top=0.96, bottom=0.16)

    fig.supylabel(r"$\Delta s_{\mathrm{cond}}$  (conditional bias)",
                  fontsize=8, x=0.005, y=0.55)
    fig.supxlabel(r"$\Delta r_{\mathrm{def}}$  (deferral rate)",
                  fontsize=8, x=0.50, y=0.02)

    # Vertical family legend on the right. Two-pass: draw rotated text,
    # measure its bbox, then place the marker swatch a fixed gap above
    # so each (swatch, label) pair has consistent spacing.
    x_col     = 0.96
    swatch_d  = 0.020
    intra_gap = 0.020
    inter_gap = 0.035

    text_objs = []
    y_cursor = 0.18
    for fam in FAMILY_ORDER:
        t = fig.text(x_col, y_cursor, FAMILY_DISPLAY[fam],
                     rotation=90, ha="center", va="bottom", fontsize=8)
        text_objs.append(t)
        y_cursor += 0.18

    fig.canvas.draw()
    inv = fig.transFigure.inverted()

    y_cursor = 0.18
    for t, fam in zip(text_objs, FAMILY_ORDER, strict=True):
        t.set_y(y_cursor)
        fig.canvas.draw()
        bbox = t.get_window_extent().transformed(inv)
        y_swatch = bbox.y1 + intra_gap + swatch_d / 2
        marker = mlines.Line2D(
            [x_col], [y_swatch],
            transform=fig.transFigure,
            marker="o", markersize=5,
            markerfacecolor=FAMILY_COLOURS[fam],
            markeredgecolor="white", markeredgewidth=0.6, linestyle="none",
        )
        fig.add_artist(marker)
        y_cursor = y_swatch + swatch_d / 2 + inter_gap

    out_png = FIGS / "bbq_quadrant.png"
    out_pdf = FIGS / "bbq_quadrant.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")
    return out_pdf


def main():
    df = load()
    print(f"loaded {len(df)} pairs")
    for quad, mask in [
        ("top-right (more def, more cond)", (df.delta_def > 0) & (df.delta_cond > 0)),
        ("top-left  (less def, more cond)", (df.delta_def < 0) & (df.delta_cond > 0)),
        ("bot-right (more def, less cond)", (df.delta_def > 0) & (df.delta_cond < 0)),
        ("bot-left  (less def, less cond)", (df.delta_def < 0) & (df.delta_cond < 0)),
    ]:
        print(f"  {quad}:  {int(mask.sum())}/{len(df)}")
    make_figure(df)


if __name__ == "__main__":
    main()
