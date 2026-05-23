"""Cached dataset loaders.

Centralises the workarounds for benchmarks whose canonical HF datasets use
loading scripts (unsupported by datasets>=3.0): CrowS-Pairs via the NYU MLL
GitHub CSV, BBQ via the `oskarvanderwal/bbq` script-free mirror, and
StereoSet via local flattening of its struct schema.

Downloads cache under ~/.cache/biaseval/ unless BIASEVAL_CACHE is set.
"""

from __future__ import annotations

import csv
import logging
import os
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_ROOT = Path(os.environ.get("BIASEVAL_CACHE", Path.home() / ".cache" / "biaseval"))

CROWS_PAIRS_URL = (
    "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"
)

# Bai et al. 2025 IAT stimuli, MIT licensed (see upstream repo).
IAT_STIMULI_URL = (
    "https://raw.githubusercontent.com/baixuechunzi/llm-implicit-bias/main/stimuli/iat_stimuli.csv"
)

# Human-readable target labels per (category, dataset). Pattern 1 tests use
# upstream A/B column names; Pattern 2 tests (paired names) get a conceptual label.
IAT_TARGET_LABELS: dict[tuple[str, str], tuple[str, str]] = {
    ("race",     "arab/muslim"):    ("european_names", "arab_muslim_names"),
    ("race",     "asian"):          ("white_surnames", "asian_surnames"),
    ("race",     "black"):          ("white_surnames", "black_surnames"),
    ("race",     "hispanic"):       ("white_surnames", "hispanic_surnames"),
    ("race",     "english"):        ("english_proficient_label", "english_learner_label"),
    ("gender",   "career"):         ("male_names",    "female_names"),
    ("gender",   "power"):          ("male_names",    "female_names"),
    ("gender",   "science"):        ("male_label",    "female_label"),
    ("gender",   "sexuality"):      ("straight_label", "gay_label"),
    ("religion", "buddhism"):       ("christian_label", "buddhist_label"),
    ("religion", "islam"):          ("christian_label", "muslim_label"),
    ("religion", "judaism"):        ("christian_label", "jewish_label"),
    ("age",      "age"):            ("young_label",   "old_label"),
    ("health",   "disability"):     ("abled_label",   "disabled_label"),
    ("health",   "eating"):         ("thin_label",    "fat_label"),
    ("health",   "mental illness"): ("physical_illness_label", "mental_illness_label"),
    ("health",   "weight"):         ("thin_label",    "fat_label"),
}

# Attribute side labels: attr_a is stereotype-aligned with target_a (canonical
# IAT pairing). In the upstream CSV, the first half of the C column is attr_a,
# the second half is attr_b.
IAT_ATTR_LABELS: dict[tuple[str, str], tuple[str, str]] = {
    ("race",     "racism"):         ("pleasant",  "unpleasant"),
    ("race",     "weapon"):         ("harmless",  "weapons"),
    ("race",     "guilt"):          ("acquittal", "conviction"),
    ("race",     "skintone"):       ("attractive", "unattractive"),
    ("race",     "arab/muslim"):    ("pleasant",  "unpleasant"),
    ("race",     "asian"):          ("pleasant",  "unpleasant"),
    ("race",     "black"):          ("pleasant",  "unpleasant"),
    ("race",     "hispanic"):       ("pleasant",  "unpleasant"),
    ("race",     "english"):        ("pleasant",  "unpleasant"),
    ("gender",   "career"):         ("career",    "family"),
    ("gender",   "power"):          ("strong",    "weak"),
    ("gender",   "science"):        ("science",   "liberal_arts"),
    ("gender",   "sexuality"):      ("good",      "bad"),
    ("religion", "buddhism"):       ("pleasant",  "unpleasant"),
    ("religion", "islam"):          ("pleasant",  "unpleasant"),
    ("religion", "judaism"):        ("pleasant",  "unpleasant"),
    ("age",      "age"):            ("pleasant",  "unpleasant"),
    ("health",   "disability"):     ("good",      "bad"),
    ("health",   "eating"):         ("good",      "bad"),
    ("health",   "mental illness"): ("temporary", "permanent"),
    ("health",   "weight"):         ("good",      "bad"),
}


def _ensure_cache(name: str) -> Path:
    p = CACHE_ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def fetch_crows_pairs() -> list[dict]:
    """Download and cache CrowS-Pairs (Nangia et al. 2020).

    Each row has at least `sent_more`, `sent_less`, `bias_type`,
    `stereo_antistereo`.
    """
    cache = _ensure_cache("crows_pairs.csv")
    if not cache.exists():
        logger.info("Downloading CrowS-Pairs to %s", cache)
        urllib.request.urlretrieve(CROWS_PAIRS_URL, cache)
    with open(cache, newline="") as f:
        rows = list(csv.DictReader(f))
    logger.info("CrowS-Pairs: %d rows", len(rows))
    return rows


def load_stereoset_intrasentence(split: str = "validation") -> list[dict]:
    """Load + flatten StereoSet (intrasentence) into one row per sentence triple.

    Each output dict has: id, context, bias_type, sentences=[{sentence, gold_label}, ...].
    """
    from datasets import load_dataset

    ds = load_dataset("McGill-NLP/stereoset", "intrasentence", split=split)
    out: list[dict] = []
    for row in ds:
        # `sentences` is a struct-of-lists; flatten to one dict per candidate.
        sents = row["sentences"]
        triples = [
            {"sentence": sents["sentence"][i], "gold_label": int(sents["gold_label"][i])}
            for i in range(len(sents["sentence"]))
        ]
        out.append({
            "id": row.get("id"),
            "context": row["context"],
            "bias_type": row["bias_type"],
            "sentences": triples,
        })
    return out


def load_bbq(split: str = "test") -> list[dict]:
    """Load BBQ from the script-free mirror at oskarvanderwal/bbq."""
    from datasets import load_dataset

    ds = load_dataset("oskarvanderwal/bbq", split=split)
    return [dict(row) for row in ds]


def load_iat_stimuli() -> list[dict]:
    """Load Bai et al. (2025) IAT stimuli into the test-dict format
    `benchmarks.iat.run` consumes.

    The upstream CSV groups rows by (category, dataset). Pattern 2 tests
    (paired names) put one target per row in columns A and B; Pattern 1 tests
    (concept words) use a single A/B header. Column C holds the attribute
    words, split 50/50: first half stereotype-aligned with target_a (attr_a),
    second half with target_b (attr_b).
    """
    cache = _ensure_cache("bai_iat_stimuli.csv")
    if not cache.exists():
        logger.info("Downloading Bai et al. IAT stimuli to %s", cache)
        urllib.request.urlretrieve(IAT_STIMULI_URL, cache)

    grouped: dict[tuple[str, str], dict[str, list[str]]] = {}
    with open(cache, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["category"].strip(), row["dataset"].strip())
            g = grouped.setdefault(key, {"A": [], "B": [], "C": []})
            a, b, c = row["A"].strip(), row["B"].strip(), row["C"].strip()
            if a: g["A"].append(a)
            if b: g["B"].append(b)
            if c: g["C"].append(c)

    tests: list[dict] = []
    for (category, subcategory), g in grouped.items():
        if not g["A"] or not g["B"] or not g["C"]:
            logger.warning("Skipping IAT test %s/%s: incomplete columns", category, subcategory)
            continue
        half = len(g["C"]) // 2
        if len(g["C"]) % 2:
            logger.warning(
                "IAT test %s/%s has odd # attribute words (%d); truncating",
                category, subcategory, len(g["C"]),
            )
        attr_a_stim = g["C"][:half]
        attr_b_stim = g["C"][half : 2 * half]

        ta_label, tb_label = IAT_TARGET_LABELS.get(
            (category, subcategory), (g["A"][0], g["B"][0]),
        )
        aa_label, ab_label = IAT_ATTR_LABELS.get(
            (category, subcategory), ("attr_a", "attr_b"),
        )

        tests.append({
            "category": category,
            "subcategory": subcategory,
            "target_a": {"name": ta_label, "stimuli": list(g["A"])},
            "target_b": {"name": tb_label, "stimuli": list(g["B"])},
            "attr_a":   {"name": aa_label, "stimuli": attr_a_stim},
            "attr_b":   {"name": ab_label, "stimuli": attr_b_stim},
        })
    logger.info(
        "Loaded %d Bai et al. IAT tests across %d categories",
        len(tests), len({t["category"] for t in tests}),
    )
    return tests
