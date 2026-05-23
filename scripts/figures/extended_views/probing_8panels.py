"""Extended view — gender-probing accuracy curves, one panel per probed pair.

The paper's Figure 5 shows one representative pair per family in a 2×2
grid. This script renders all 8 probed pairs in a 2×4 grid so the
"curves overlap in 7 of 8 pairs" claim can be verified visually. Not
included in the paper itself.

    solid line  : base model     (family colour)
    dashed line : instruct model (family colour, same hue)

Dotted gray line at 0.5 marks chance. The visible exceptions to the
overlap claim are Llama 2 13B and Gemma 3 4B (the latter excluded from
the paper figure as an anomaly).

Saves figures/extended_views/probing_8panels.{png,pdf}.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
os.chdir(REPO)

FIGS = Path("figures/extended_views")
FIGS.mkdir(parents=True, exist_ok=True)

FAMILY_COLOURS = {
    "llama":   "#1f77b4",
    "qwen":    "#ff7f0e",
    "mistral": "#d62728",
    "gemma":   "#2ca02c",
}

# (family, generation, size, short_label) in display order.
PANELS = [
    ("llama",   "Llama 2",         "7B",  "Llama 2, 7B"),
    ("llama",   "Llama 2",         "13B", "Llama 2, 13B"),
    ("llama",   "Llama 3.1",       "8B",  "Llama 3.1, 8B"),
    ("qwen",    "Qwen 2.5",        "7B",  "Qwen 2.5, 7B"),
    ("mistral", "Mistral v0.3",    "7B",  "Mistral v0.3, 7B"),
    ("mistral", "Mistral Small 3", "24B", "Mistral Small 3, 24B"),
    ("gemma",   "Gemma 2",         "9B",  "Gemma 2, 9B"),
    ("gemma",   "Gemma 3",         "4B",  "Gemma 3, 4B"),
]

Y_LO, Y_HI = 0.45, 0.90


def _draw_panel(ax, df: pd.DataFrame, fam: str, gen: str, size: str, title: str):
    sub = df[(df.family == fam) & (df.generation == gen) & (df["size"] == size)]
    base = sub[sub.variant == "base"].sort_values("layer_normalized")
    inst = sub[sub.variant == "instruct"].sort_values("layer_normalized")
    c = FAMILY_COLOURS[fam]

    # Chance line — soft gray but thick enough to read clearly behind data.
    ax.axhline(0.5, color="#a8a8a8", linestyle=":", linewidth=1.4, zorder=1)

    ax.plot(base["layer_normalized"], base["mean_accuracy"],
            color=c, linestyle="-", linewidth=1.6, zorder=3, label="base")
    ax.plot(inst["layer_normalized"], inst["mean_accuracy"],
            color=c, linestyle="--", linewidth=1.6, zorder=3, label="instruct")

    ax.set_title(title, fontsize=9.5, pad=3)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(Y_LO, Y_HI)
    # Tick labels: subtly smaller and dark-gray so the data lines lead the eye.
    ax.tick_params(axis="both", which="both",
                   labelsize=7.5, labelcolor="#555555", length=2.5)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="-", linewidth=0.4, color="#eeeeee", zorder=0)


def make_figure() -> Path:
    df = pd.read_csv("data/tables/probe_per_layer.csv")
    df = df[df.attribute == "gender"].copy()

    fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.5),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for ax, panel in zip(axes_flat, PANELS, strict=True):
        if panel is None:
            ax.set_visible(False)
            continue
        fam, gen, size, title = panel
        _draw_panel(ax, df, fam, gen, size, title)

    # Tufte-style chrome cleanup: tick numbers + tick marks only on the
    # outermost panels.  Inner panels keep the spine and gridlines but lose
    # all redundant axis text.
    for i in range(2):
        for j in range(4):
            ax = axes[i, j]
            if j != 0:
                ax.tick_params(axis="y", labelleft=False, left=False)
            if i != 1:
                ax.tick_params(axis="x", labelbottom=False, bottom=False)

# Strip per-axis labels so the figure-level supylabel / supxlabel act as
    # the single shared "frame".
    for ax in axes.flatten():
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.supylabel("Probe accuracy", fontsize=12, x=0.045, y=0.56)
    fig.supxlabel("Normalised layer depth", fontsize=12, y=0.10)

    # Shared legend at the bottom — two style entries, family-agnostic.
    handles = [
        mlines.Line2D([], [], color="#444", linestyle="-",  linewidth=1.6, label="base"),
        mlines.Line2D([], [], color="#444", linestyle="--", linewidth=1.6, label="instruct"),
        mlines.Line2D([], [], color="#a8a8a8", linestyle=":", linewidth=1.4,
                      label="chance (0.5)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04),
               frameon=False, fontsize=9, handletextpad=0.6, columnspacing=2.5)

    fig.tight_layout(rect=[0.03, 0.08, 1.0, 1.0])
    # Horizontal breathing room so the 0.0/1.0 tick labels don't crowd
    # their neighbours.
    fig.subplots_adjust(wspace=0.15)

    out_png = FIGS / "probing_8panels.png"
    out_pdf = FIGS / "probing_8panels.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight",
                transparent=False, facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight",
                transparent=False, facecolor="white")
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")
    return out_pdf


if __name__ == "__main__":
    make_figure()
