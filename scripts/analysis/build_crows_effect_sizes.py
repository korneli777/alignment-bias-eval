"""Build exact paired CrowS-Pairs effect sizes from per-example results.

The output contains paired outcome counts and Cohen's d, but no benchmark
sentences or model log probabilities. Both paper conditions are included:
base/plain versus instruct/plain, and base/plain versus instruct/chat.

Usage:
    python scripts/analysis/build_crows_effect_sizes.py \
        --data-root /path/to/data
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import pandas as pd
import yaml

from biaseval.analysis.statistics import paired_binary_effect_summary

logger = logging.getLogger(__name__)


def paired_effect_size(base: dict[int, bool], instruct: dict[int, bool]) -> dict:
    """Sufficient counts and paired Cohen's d for two outcome vectors."""
    if base.keys() != instruct.keys():
        missing_base = sorted(instruct.keys() - base.keys())
        missing_instruct = sorted(base.keys() - instruct.keys())
        raise ValueError(
            "outcome IDs differ: "
            f"missing from base={missing_base[:5]}, "
            f"missing from instruct={missing_instruct[:5]}"
        )
    item_ids = sorted(base)
    summary = paired_binary_effect_summary(
        [base[item_id] for item_id in item_ids],
        [instruct[item_id] for item_id in item_ids],
    )
    summary["n_base_stereo"] = summary.pop("n_base_positive")
    summary["n_instruct_stereo"] = summary.pop("n_instruct_positive")
    return summary


def load_outcomes(path: Path, expected_mode: str) -> dict[int, bool]:
    """Load pair IDs and normalized CrowS outcomes from one result JSON."""
    payload = json.loads(path.read_text())
    result = payload["result"]
    if result["benchmark"] != "crows_pairs":
        raise ValueError(f"not a CrowS-Pairs result: {path}")
    if result["prompt_mode"] != expected_mode:
        raise ValueError(
            f"expected prompt_mode={expected_mode!r}, found "
            f"{result['prompt_mode']!r}: {path}"
        )
    outcomes = {
        int(row["pair_id"]): bool(row["stereo_won"])
        for row in result["per_example"]
    }
    if len(outcomes) != len(result["per_example"]):
        raise ValueError(f"duplicate pair_id in {path}")
    return outcomes


def registry_pairs(config_path: Path) -> list[dict]:
    """Base-instruct pairs and their display metadata from the model registry."""
    config = yaml.safe_load(config_path.read_text())
    return [
        {
            "family": family_name,
            "generation": generation["name"],
            "size": model["size"],
            "base_id": model["base_id"],
            "instruct_id": model["instruct_id"],
        }
        for family_name, family in config["families"].items()
        for generation in family["generations"]
        for model in generation["models"]
    ]


def build_rows(data_root: Path, config_path: Path) -> pd.DataFrame:
    """Build both scoring conditions for every registered model pair."""
    source = data_root / "raw_logit_scores" / "crows_pairs"
    missing: list[Path] = []
    rows: list[dict] = []

    for pair in registry_pairs(config_path):
        base_path = source / f"{pair['base_id'].replace('/', '__')}__raw.json"
        condition_paths = {
            "without_template": source / f"{pair['instruct_id'].replace('/', '__')}__raw.json",
            "with_template": source / f"{pair['instruct_id'].replace('/', '__')}__instruct.json",
        }
        needed = [base_path, *condition_paths.values()]
        if absent := [path for path in needed if not path.is_file()]:
            missing.extend(absent)
            continue

        base = load_outcomes(base_path, "raw")
        for condition, instruct_path in condition_paths.items():
            prompt_mode = "raw" if condition == "without_template" else "instruct"
            instruct = load_outcomes(instruct_path, prompt_mode)
            rows.append({
                **pair,
                "condition": condition,
                **paired_effect_size(base, instruct),
            })

    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:10])
        suffix = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"missing {len(missing)} required CrowS-Pairs result files:\n{preview}{suffix}"
        )

    return pd.DataFrame(rows)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write the derived table without leaving a partial output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/tables/crows_pair_effect_sizes.csv")
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        rows = build_rows(args.data_root, args.config)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        logger.error("%s", error)
        return 1

    write_csv_atomic(rows, args.output)
    logger.info(
        "Wrote %d model-condition rows to %s (max |d| = %.3f)",
        len(rows), args.output, rows["cohens_d"].abs().max(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
