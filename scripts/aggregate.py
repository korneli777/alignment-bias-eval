"""Aggregate raw per-cell result JSONs into the three parquet sheets.

Reads from:
    data/raw_logit_scores/<benchmark>/*.json
    data/probe_results/<model>/*.json
    data/intervention_results/<benchmark>/*.json

Writes to:
    data/aggregated/logit.parquet
    data/aggregated/probe.parquet
    data/aggregated/intervention.parquet

Run this after a fresh GPU pipeline (or after downloading raw JSONs from
the Zenodo deposit) to regenerate the parquet sheets that the analysis
scripts and figure scripts read.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from biaseval.analysis.aggregate import write_aggregated

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-root", default="data",
                   help="Directory containing raw_logit_scores/, probe_results/, "
                        "intervention_results/. Default: data")
    p.add_argument("--out-dir", default="data/aggregated",
                   help="Where to write the three parquets. Default: data/aggregated")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    counts = write_aggregated(Path(args.data_root), Path(args.out_dir))
    logger.info(
        "Wrote logit=%d rows, probe=%d rows, intervention=%d rows",
        counts["logit_rows"], counts["probe_rows"], counts["intervention_rows"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
