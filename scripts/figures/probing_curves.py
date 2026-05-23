"""2x2 grid of per-layer gender-probing accuracy curves, one panel per family,
size-controlled to the 7-9B band. The full 8-pair grid lives in
scripts/figures/extended_views/probing_8panels.py.
Saves figures/probing_curves.{png,pdf}. (Paper Figure 5.)
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)

FIGS = Path("figures")
FIGS.mkdir(parents=True, exist_ok=True)

FAMILY_COLOURS = {
    "llama":   "#1f77b4",
    "qwen":    "#ff7f0e",
    "mistral": "#d62728",
    "gemma":   "#2ca02c",
}

PANELS = [
    ("llama",   "Llama 3.1",    "8B", "Llama 3.1, 8B"),
    ("qwen",    "Qwen 2.5",     "7B", "Qwen 2.5, 7B"),
    ("mistral", "Mistral v0.3", "7B", "Mistral v0.3, 7B"),
    ("gemma",   "Gemma 2",      "9B", "Gemma 2, 9B"),
]

Y_LO, Y_HI = 0.45, 0.90


def _draw_panel(ax, df: pd.DataFrame, fam: str, gen: str, size: str, title: str):
    sub = df[(df.family == fam) & (df.generation == gen) & (df["size"] == size)]
    base = sub[sub.variant == "base"].sort_values("layer_normalized")
    inst = sub[sub.variant == "instruct"].sort_values("layer_normalized")
    c = FAMILY_COLOURS[fam]

    ax.axhline(0.5, color="#a8a8a8", linestyle=":", linewidth=1.2, zorder=1)
    ax.plot(base["layer_normalized"], base["mean_accuracy"],
            color=c, linestyle="-",  linewidth=1.5, zorder=3)
    ax.plot(inst["layer_normalized"], inst["mean_accuracy"],
            color=c, linestyle="--", linewidth=1.5, zorder=3)

    ax.set_title(title, fontsize=8, pad=2.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(Y_LO, Y_HI)
    ax.tick_params(axis="both", labelsize=7, labelcolor="#555555", length=0)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    for sp in ax.spines.values():
        sp.set_visible(False)
    # Skip a gridline at 0.5; the chance reference already covers it.
    for y_grid in (0.6, 0.7, 0.8, 0.9):
        ax.axhline(y_grid, color="#eeeeee", linewidth=0.4, zorder=0)


def make_figure() -> Path:
    df = pd.read_csv("data/tables/probe_per_layer.csv")
    df = df[df.attribute == "gender"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.4),
                             sharex=True, sharey=True)

    for ax, (fam, gen, size, title) in zip(axes.flatten(), PANELS, strict=True):
        _draw_panel(ax, df, fam, gen, size, title)

    # Numbers only on the left column and bottom row.
    for i in range(2):
        for j in range(2):
            ax = axes[i, j]
            if j != 0:
                ax.tick_params(axis="y", labelleft=False, left=False)
            if i != 1:
                ax.tick_params(axis="x", labelbottom=False, bottom=False)

    for ax in axes.flatten():
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.subplots_adjust(left=0.16, right=0.84, top=0.96, bottom=0.13,
                        wspace=0.20, hspace=0.36)
    fig.supylabel("Probe accuracy",        fontsize=8, x=0.005, y=0.54)
    fig.supxlabel("Normalised layer depth", fontsize=8, x=0.50,  y=0.005)

    # Vertical legend rail mirroring the rotated y-label on the left.
    # Two-pass: draw rotated labels, measure their bboxes, then place
    # each line sample a fixed gap above its label so spacing is uniform
    # regardless of font metrics.
    legend_entries = [
        # (label, linestyle, colour, linewidth, italic)
        ("base",     "-",  "#444",    1.8, False),
        ("instruct", "--", "#444",    1.8, False),
        ("chance",   ":",  "#7a7a7a", 2.6, True),
    ]
    x_col     = 0.96
    line_h    = 0.11
    intra_gap = 0.020
    inter_gap = 0.045

    text_objs = []
    y_cursor = 0.16
    for label, ls, color, lw, italic in legend_entries:
        t = fig.text(
            x_col, y_cursor, label,
            rotation=90, ha="center", va="bottom",
            fontsize=8,
            fontstyle="italic" if italic else "normal",
        )
        text_objs.append((t, ls, color, lw))
        y_cursor += 0.22

    fig.canvas.draw()
    inv = fig.transFigure.inverted()

    y_cursor = 0.16
    for (t, ls, color, lw) in text_objs:
        t.set_y(y_cursor)
        fig.canvas.draw()
        bbox = t.get_window_extent().transformed(inv)
        y_line_bot = bbox.y1 + intra_gap
        y_line_top = y_line_bot + line_h
        sample = mlines.Line2D(
            [x_col, x_col], [y_line_bot, y_line_top],
            transform=fig.transFigure,
            color=color, linestyle=ls, linewidth=lw,
            solid_capstyle="butt",
        )
        fig.add_artist(sample)
        y_cursor = y_line_top + inter_gap

    out_png = FIGS / "probing_curves.png"
    out_pdf = FIGS / "probing_curves.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")
    return out_pdf


if __name__ == "__main__":
    make_figure()
