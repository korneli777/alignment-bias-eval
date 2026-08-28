"""Per-pair dumbbell of CrowS-Pairs and StereoSet under raw vs chat-template
scoring. Saves figures/chat_template_dumbbell.{png,pdf}. (Paper Figure 3.)
"""
from __future__ import annotations

import itertools
import os
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)

FIGS = Path("figures")
FIGS.mkdir(parents=True, exist_ok=True)

METRIC = {"crows_pairs": "overall", "stereoset": "overall_SS"}
TITLE  = {"crows_pairs": "CrowS-Pairs", "stereoset": "StereoSet"}

FAMILY_ORDER = ["llama", "qwen", "mistral", "gemma"]
FAMILY_COLOURS = {
    "llama":   "#1f77b4",
    "qwen":    "#ff7f0e",
    "mistral": "#d62728",
    "gemma":   "#2ca02c",
}
FAMILY_LABELS = {
    "llama": "Llama",
    "qwen": "Qwen",
    "mistral": "Mistral",
    "gemma": "Gemma",
}


def load() -> pd.DataFrame:
    df = pd.read_parquet("data/aggregated/logit.parquet")
    rows = []
    for bench, met in METRIC.items():
        sub = df[(df["benchmark"] == bench) & (df["metric"] == met)
                 & (df["prompt_mode"].isin(["raw", "instruct"]))].copy()
        rows.append(sub)
    h = pd.concat(rows, ignore_index=True)
    pivot = h.pivot_table(
        index=["family", "generation", "size", "num_params", "benchmark"],
        columns=["variant", "prompt_mode"], values="value", aggfunc="first",
    )
    keep = []
    for keys, row in pivot.iterrows():
        fam, gen, size, num_params, bench = keys
        b_r = row.get(("base", "raw"))
        i_r = row.get(("instruct", "raw"))
        i_c = row.get(("instruct", "instruct"))
        if any(pd.isna([b_r, i_r, i_c])):
            continue
        keep.append({
            "family": fam, "generation": gen, "size": size,
            "num_params": int(num_params), "benchmark": bench,
            "raw_delta":    i_r - b_r,
            "native_delta": i_c - b_r,
        })
    return pd.DataFrame(keep)


def _generation_dates() -> dict[tuple[str, str], str]:
    with open("configs/models.yaml") as f:
        cfg = yaml.safe_load(f)
    return {
        (fam, gen["name"]): gen["release_date"]
        for fam, family in cfg["families"].items()
        for gen in family["generations"]
    }


def _layout(df: pd.DataFrame, *, gap: float = 0.9, mean_row_gap: float = 1.8):
    union = (
        df[["family", "generation", "size", "num_params"]]
        .drop_duplicates()
        .copy()
    )
    dates = _generation_dates()
    union["gen_date"] = union.apply(
        lambda r: dates.get((r["family"], r["generation"]), "9999-99"), axis=1,
    )
    y = 0.0
    pos, family_mean_pos, group_top, group_bottom = {}, {}, {}, {}
    for fam in FAMILY_ORDER:
        members = union[union["family"] == fam].sort_values(
            by=["gen_date", "num_params"], ascending=[True, False],
        )
        if members.empty:
            continue
        family_mean_pos[fam] = y
        group_top[fam] = y
        y += 1.0
        ys = []
        for _, r in members.iterrows():
            pos[(fam, r["generation"], r["size"])] = y
            ys.append(y)
            y += 1.0
        group_bottom[fam] = ys[-1]
        y += gap
    return pos, family_mean_pos, -mean_row_gap, group_top, group_bottom


def _bootstrap_mean_ci(
    values: pd.Series, rng: np.random.Generator, n_resamples: int = 10_000,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a mean."""
    x = values.to_numpy(dtype=float)
    sampled = x[rng.integers(0, len(x), size=(n_resamples, len(x)))]
    lo, hi = np.quantile(sampled.mean(axis=1), [0.025, 0.975])
    return float(lo), float(hi)


def _draw_mean_row(
    ax,
    sub: pd.DataFrame,
    y: float,
    colour: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Draw one overall or family mean row with bootstrap bands."""
    raw_mean = float(sub["raw_delta"].mean())
    native_mean = float(sub["native_delta"].mean())
    raw_lo, raw_hi = _bootstrap_mean_ci(sub["raw_delta"], rng)
    native_lo, native_hi = _bootstrap_mean_ci(sub["native_delta"], rng)

    band_y = [y - 0.19, y + 0.19]
    ax.fill_betweenx(band_y, raw_lo, raw_hi, color=colour, alpha=0.13, linewidth=0, zorder=1)
    ax.fill_betweenx(
        band_y, native_lo, native_hi, color=colour, alpha=0.13, linewidth=0, zorder=1,
    )
    ax.plot(
        [raw_mean, native_mean], [y, y], color="#c9c9c9", linewidth=2.0,
        solid_capstyle="butt", zorder=2,
    )
    ax.scatter(
        raw_mean, y, s=70, marker="o", facecolors="white", edgecolors=colour,
        linewidth=2.0, zorder=7,
    )
    ax.scatter(
        native_mean, y, s=100, marker="o", facecolors=colour, edgecolors="white",
        linewidth=0.9, zorder=8,
    )
    return raw_mean, native_mean


def _draw_panel(ax, sub: pd.DataFrame, bench: str, pos: dict,
                family_mean_pos: dict, mean_y: float, group_top: dict,
                group_bottom: dict, *,
                show_yticks: bool,
                xlim: tuple[float, float],
                xticks: list[float]):
    rng = np.random.default_rng(42)

    # Background bands so the reader sees "left = better, right = worse"
    # before reading any labels.
    ax.axvspan(-1e6, 0, color="#edf7ed", zorder=0)
    ax.axvspan(0, 1e6,  color="#fdf0f0", zorder=0)

    for yv in sorted([*pos.values(), *family_mean_pos.values()]):
        ax.axhline(yv, color="#ebebeb", linewidth=0.6, zorder=0)
    ax.axhline(mean_y, color="#e1e1e1", linewidth=0.6, zorder=0)

    fam_keys = list(group_top.keys())
    for f1, f2 in itertools.pairwise(fam_keys):
        ax.axhline((group_bottom[f1] + group_top[f2]) / 2.0,
                   color="#d4d4d4", linewidth=0.8, zorder=1)
    ax.axvline(0, color="black", linewidth=1.5, zorder=4)

    ax.text(0.02, 0.985, "← Alignment reduces\n   stereotype score",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=11, color="#3a703a", style="italic", linespacing=1.25)
    ax.text(0.98, 0.985, "Alignment increases\nstereotype score →",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, color="#a13a3a", style="italic", linespacing=1.25)

    mean_gray = "#777777"
    raw_mean, native_mean = _draw_mean_row(ax, sub, mean_y, mean_gray, rng)
    ax.axvline(raw_mean, color=mean_gray, linestyle="--", linewidth=1.0, zorder=3)
    ax.axvline(native_mean, color=mean_gray, linestyle="--", linewidth=1.4, zorder=3)

    for fam in FAMILY_ORDER:
        if fam not in family_mean_pos:
            continue
        family_rows = sub[sub["family"] == fam]
        if not family_rows.empty:
            _draw_mean_row(
                ax, family_rows, family_mean_pos[fam], FAMILY_COLOURS[fam], rng,
            )

    for _, r in sub.iterrows():
        key = (r["family"], r["generation"], r["size"])
        if key not in pos:
            continue
        y = pos[key]
        c = FAMILY_COLOURS.get(r["family"], "grey")
        ax.plot([r["raw_delta"], r["native_delta"]], [y, y],
                color="#c9c9c9", linewidth=2.0, solid_capstyle="butt", zorder=2)
        ax.scatter(r["raw_delta"], y, s=80, marker="o",
                   facecolors="white", edgecolors=c, linewidth=2.0, zorder=5)
        ax.scatter(r["native_delta"], y, s=120, marker="o",
                   facecolors=c, edgecolors="white", linewidth=0.9, zorder=6)

    used_pairs = sorted(pos.items(), key=lambda kv: kv[1])
    if show_yticks:
        family_rows = [
            (family_mean_pos[fam], f"{FAMILY_LABELS[fam]} Mean", fam)
            for fam in FAMILY_ORDER if fam in family_mean_pos
        ]
        model_rows = [
            (yv, f"{gen} {size}", fam)
            for (fam, gen, size), yv in used_pairs
        ]
        labelled_rows = sorted([*family_rows, *model_rows], key=lambda row: row[0])
        ax.set_yticks([mean_y] + [row[0] for row in labelled_rows])
        labels = ["Mean"] + [row[1] for row in labelled_rows]
        ax.set_yticklabels(labels, fontsize=13)
        colours = [mean_gray] + [FAMILY_COLOURS[row[2]] for row in labelled_rows]
        family_ys = set(family_mean_pos.values())
        weights = ["bold"] + [
            "bold" if row[0] in family_ys else "normal" for row in labelled_rows
        ]
        for tick, col, w in zip(ax.get_yticklabels(), colours, weights, strict=True):
            tick.set_color(col)
            tick.set_fontweight(w)
    else:
        ax.set_yticks([])

    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    y_max = max(group_bottom.values())
    ax.set_ylim(mean_y - 2.6, y_max + 0.7)
    ax.invert_yaxis()
    ax.set_xlabel("")
    ax.set_title(TITLE[bench], fontsize=18, pad=46, fontweight="bold")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(axis="x", length=4, labelsize=13)
    ax.tick_params(axis="y", length=4)


def make_figure(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 12.0))
    pos, family_mean_pos, mean_y, g_top, g_bot = _layout(df)

    xlim   = (-15.0, 6.0)
    xticks = [-15, -10, -5, 0, 5]

    for i, (ax, bench) in enumerate(zip(axes, ["stereoset", "crows_pairs"], strict=True)):
        sub = df[df["benchmark"] == bench]
        if sub.empty:
            ax.set_visible(False)
            continue
        _draw_panel(ax, sub, bench, pos=pos, family_mean_pos=family_mean_pos,
                    mean_y=mean_y, group_top=g_top, group_bottom=g_bot,
                    show_yticks=(i == 0), xlim=xlim, xticks=xticks)

    fig.supxlabel("Change in stereotype score from base to instruct (pp)",
                  fontsize=15, y=0.04)

    shape_handles = [
        mlines.Line2D([], [], color="black", marker="o", linestyle="none",
                      markersize=14, markerfacecolor="black",
                      markeredgecolor="white", markeredgewidth=0.6,
                      label="with chat template"),
        mlines.Line2D([], [], color="black", marker="o", linestyle="none",
                      markersize=14, markerfacecolor="white",
                      markeredgewidth=2.0,
                      label="without chat template"),
    ]
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.94])

    bbox = axes[0].get_position()
    title_pad_fig = 46 / 72 / fig.get_size_inches()[1]
    legend_y = bbox.y1 + title_pad_fig * 0.30
    fig.legend(handles=shape_handles,
               loc="center left", ncol=2,
               bbox_to_anchor=(bbox.x0, legend_y),
               frameon=False, fontsize=13,
               handletextpad=0.6, columnspacing=2.0)

    out_png = FIGS / "chat_template_dumbbell.png"
    out_pdf = FIGS / "chat_template_dumbbell.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")
    return out_png


def main():
    df = load()
    print(f"loaded {len(df)} pair × benchmark rows")
    print(df.groupby("benchmark").size().to_dict())
    make_figure(df)


if __name__ == "__main__":
    main()
