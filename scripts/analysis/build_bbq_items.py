"""Build the per-item BBQ table backing the framing decomposition (Appendix A.3).

The paper reports BBQ ambiguous-context results as a deferral rate and a
conditional bias on committed answers. This script emits the per-item records
those two components are computed from, so a reader can recompute either one
without the multi-GB raw score dumps.

Every condition is put on the *same* fixed 6,001-item set. The framing runs
scored that set directly (`bbq.run(..., ambiguous_only=True, subsample=6000)`).
The value 6000 is the proportional category budget; independent rounding of
the category allocations produces 6,001 rows on the released BBQ split.
The base and instruct baselines were scored on the full split, so their records
are re-filtered here by reproducing the same seeded, category-stratified
selection. Before writing, the script verifies that every condition has the
same item identifiers, categories, and question polarities. Without that
check, the comparison could straddle different item sets.

Usage:
    python scripts/analysis/build_bbq_items.py \
        --baseline-dir  /path/to/full-split/bbq \
        --framing-dir   /path/to/framing/bbq \
        --framing-dir   /path/to/bbq_jb_academic \
        --out data/aggregated/bbq_items.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from biaseval.benchmarks.framings import BBQ_FRAMINGS

logger = logging.getLogger(__name__)

# Kept in the emitted table so each row is self-describing.
FIELDS = ("category", "question_polarity", "is_unknown_pred", "is_biased_pred")


def stratified_ambiguous_ids(per_example: list[dict], n: int, seed: int) -> list[dict]:
    """Reproduce `bbq._stratified_ambiguous_sample` on stored per-example records.

    Returns the selected records in the order the framing runs scored them, so
    positional index == the framing runs' `id`.
    """
    amb = [r for r in per_example if r.get("context_condition") == "ambig"]
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in amb:
        by_cat[str(r.get("category", "unknown"))].append(r)
    total = len(amb)
    rng = random.Random(seed)
    chosen: list[dict] = []
    for cat in sorted(by_cat):
        group = by_cat[cat]
        k = round(n * len(group) / total) if total else 0
        idx = list(range(len(group)))
        rng.shuffle(idx)
        chosen.extend(group[i] for i in sorted(idx[:k]))
    return chosen


def _rows(records: list[dict], spec: dict, prompt_mode: str) -> list[dict]:
    return [
        {
            "model_id": spec["model_id"],
            "family": spec["family"],
            "size": spec["size"],
            "variant": spec["variant"],
            "prompt_mode": prompt_mode,
            "item_id": i,
            **{f: r[f] for f in FIELDS},
        }
        for i, r in enumerate(records)
    ]


def _find(dirs: list[Path], stem: str) -> Path | None:
    for d in dirs:
        fp = d / f"{stem}.json"
        if fp.exists():
            return fp
    return None


def validate_shared_item_set(frame: pd.DataFrame) -> None:
    """Raise when model-condition cells do not describe the same BBQ items."""
    signature_columns = ["item_id", "category", "question_polarity"]
    reference_key = None
    reference = None
    for key, cell in frame.groupby(["model_id", "prompt_mode"], observed=True):
        signature = (
            cell.sort_values("item_id")[signature_columns]
            .astype(str)
            .reset_index(drop=True)
        )
        if reference is None:
            reference_key = key
            reference = signature
        elif not signature.equals(reference):
            raise ValueError(
                f"BBQ item set for {key} does not match reference cell {reference_key}"
            )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--baseline-dir", required=True, type=Path,
                   help="Full-split BBQ JSONs (*__raw.json, *__instruct.json).")
    p.add_argument("--framing-dir", required=True, type=Path, action="append",
                   help="Directory of *__jb_*.json framing results. Repeatable.")
    p.add_argument("--config", default="configs/models.yaml", type=Path)
    p.add_argument("--out", default="data/aggregated/bbq_items.parquet", type=Path)
    p.add_argument("--subsample", type=int, default=6000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write a partial table when cells are missing (never use for a release).",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = yaml.safe_load(args.config.read_text())
    pairs = [
        (m["base_id"], m["instruct_id"])
        for fd in cfg["families"].values()
        for gen in fd["generations"]
        for m in gen["models"]
    ]

    rows: list[dict] = []
    missing: list[str] = []
    expected_items: int | None = None

    for base_id, instruct_id in pairs:
        # Baselines: full split, re-filtered to the fixed 6,001-item set.
        for model_id, mode in ((base_id, "raw"), (instruct_id, "instruct")):
            fp = args.baseline_dir / f"{model_id.replace('/', '__')}__{mode}.json"
            if not fp.exists():
                missing.append(f"{model_id} [{mode}]")
                continue
            d = json.loads(fp.read_text())
            sel = stratified_ambiguous_ids(
                d["result"]["per_example"], args.subsample, args.seed
            )
            if expected_items is None:
                expected_items = len(sel)
            elif len(sel) != expected_items:
                raise ValueError(
                    f"expected {expected_items:,} selected BBQ items for {model_id}, "
                    f"got {len(sel):,}"
                )
            rows += _rows(sel, d["spec"], mode)

        # Framing results already contain the selected ambiguous-item sample.
        for fid in BBQ_FRAMINGS:
            fp = _find(args.framing_dir, f"{instruct_id.replace('/', '__')}__{fid}")
            if fp is None:
                missing.append(f"{instruct_id} [{fid}]")
                continue
            d = json.loads(fp.read_text())
            records = d["result"]["per_example"]
            if expected_items is None:
                expected_items = len(records)
            elif len(records) != expected_items:
                raise ValueError(
                    f"expected {expected_items:,} BBQ items for {instruct_id} [{fid}], "
                    f"got {len(records):,}"
                )
            rows += _rows(records, d["spec"], fid)

    if missing and not args.allow_incomplete:
        logger.error("missing %d cells: %s", len(missing), ", ".join(missing))
        logger.error("refusing to write a partial release; use --allow-incomplete for diagnostics")
        return 1

    df = pd.DataFrame(rows)
    validate_shared_item_set(df)
    for col in ("model_id", "family", "size", "variant", "prompt_mode", "category",
                "question_polarity"):
        df[col] = df[col].astype("category")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False, compression="zstd")

    logger.info("Wrote %s: %d rows, %.1f MB", args.out, len(df),
                args.out.stat().st_size / 1e6)
    logger.info("cells: %d model x condition",
                df.groupby(["model_id", "prompt_mode"], observed=True).ngroups)
    if missing:
        logger.warning("missing %d cells: %s", len(missing), ", ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
