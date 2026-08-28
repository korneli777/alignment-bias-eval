"""Labelled probe datasets where the attribute is implicit in the context.

Probes use neutral templates whose demographic association comes from a
non-keyword cue (occupation for gender, first name for race), so the probe
has to learn a context-driven representation rather than a surface lookup.

Two attributes ship out of the box:
    gender   WinoBias-40 occupations (Zhao et al. 2018) under
             "The {occupation} said that".
    race     Distinctively-racial first names (Caliskan et al. 2017 / Bai
             et al. 2025) under "{Name} walked into the room".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProbeDataset:
    """Labelled sentences for training one binary attribute probe.

    `label_names` keeps the two classes in the order the integer labels use, so
    a saved direction vector can be read back with a known sign convention.
    """

    sentences: list[str]
    labels: list[int]
    label_names: tuple[str, str]
    attribute: str

    def __len__(self) -> int:
        return len(self.sentences)


# Label 0 = male-stereotyped (BLS-skewed male), label 1 = female-stereotyped.
# Order matches the canonical WinoBias splits (Zhao et al. 2018).
WINOBIAS_OCCUPATIONS_MALE = [
    "carpenter", "mechanic", "construction worker", "laborer", "driver",
    "sheriff", "mover", "developer", "farmer", "guard",
    "chief", "janitor", "lawyer", "cook", "physician",
    "CEO", "analyst", "manager", "supervisor", "salesperson",
]
WINOBIAS_OCCUPATIONS_FEMALE = [
    "attendant", "cashier", "teacher", "nurse", "assistant",
    "secretary", "auditor", "cleaner", "receptionist", "clerk",
    "counselor", "designer", "hairdresser", "writer", "housekeeper",
    "baker", "accountant", "editor", "librarian", "tailor",
]

OCCUPATION_TEMPLATE = "The {occupation} said that"


# Distinctively-racial first names from Caliskan et al. (2017) WEAT-1/WEAT-2,
# reused in Bai et al. (2025). Label 0 = European-American, 1 = African-American.
EUROPEAN_NAMES = [
    "Adam", "Chip", "Harry", "Josh", "Roger", "Alan", "Frank", "Ian",
    "Justin", "Ryan", "Andrew", "Fred", "Jack", "Matthew", "Stephen",
    "Brad", "Greg", "Jed", "Paul", "Todd",
    "Amanda", "Courtney", "Heather", "Melanie", "Sara", "Amber", "Crystal",
    "Katie", "Meredith", "Shannon", "Betsy", "Donna", "Kristin", "Nancy",
    "Stephanie", "Bobbie-Sue", "Ellen", "Lauren", "Peggy", "Sue-Ellen",
]
AFRICAN_AMERICAN_NAMES = [
    "Alonzo", "Jamel", "Lerone", "Percell", "Theo", "Alphonse", "Jerome",
    "Leroy", "Rasaan", "Torrance", "Darnell", "Lamar", "Lionel", "Rashaun",
    "Tyree", "Deion", "Lamont", "Malik", "Terrence", "Tyrone",
    "Aiesha", "Ebony", "Lakisha", "Latoya", "Tamika", "Aisha", "Imani",
    "Latanya", "Latonya", "Tanisha", "Charisse", "Jasmine", "Latisha",
    "Shaniqua", "Tashika", "Felicia", "Kenya", "Lashelle", "Sade", "Tia",
]

NAME_TEMPLATE = "{name} walked into the room"


# Surface-keyword sets used by run_probing.py with --mask-keywords to ablate
# demographic surface forms from the activation pool.
GENDER_KEYWORDS = {
    0: {"man", "men", "boy", "boys", "father", "uncle", "brother", "husband", "son", "he", "his", "him", "male"},
    1: {"woman", "women", "girl", "girls", "mother", "aunt", "sister", "wife", "daughter", "she", "her", "hers", "female"},
}
RACE_KEYWORDS = {
    0: {"white", "european", "caucasian"},
    1: {"black", "african", "african-american", "afro-american", "asian", "hispanic", "latino", "latina", "arab", "middle eastern"},
}


def _balanced(items_a: list[str], items_b: list[str], max_per_class: int | None) -> tuple[list[str], list[str]]:
    if max_per_class is None:
        return items_a, items_b
    return items_a[:max_per_class], items_b[:max_per_class]


def build_probe_dataset(
    attribute: str,
    *,
    max_per_class: int | None = None,
) -> ProbeDataset:
    """Default: neutral-prompt probe (no demographic keyword in the sentence)."""
    if attribute == "gender":
        male_occ, female_occ = _balanced(
            WINOBIAS_OCCUPATIONS_MALE, WINOBIAS_OCCUPATIONS_FEMALE, max_per_class,
        )
        sents = [OCCUPATION_TEMPLATE.format(occupation=o) for o in male_occ + female_occ]
        labels = [0] * len(male_occ) + [1] * len(female_occ)
        names = ("male_stereotyped", "female_stereotyped")
    elif attribute == "race":
        eur, afr = _balanced(EUROPEAN_NAMES, AFRICAN_AMERICAN_NAMES, max_per_class)
        sents = [NAME_TEMPLATE.format(name=n) for n in eur + afr]
        labels = [0] * len(eur) + [1] * len(afr)
        names = ("european_american", "african_american")
    else:
        raise ValueError(f"Unknown attribute: {attribute}")

    logger.info(
        "Built neutral-prompt probe dataset for %s: %d sentences (balanced %d/%d)",
        attribute, len(sents), labels.count(0), labels.count(1),
    )
    return ProbeDataset(sentences=sents, labels=labels, label_names=names, attribute=attribute)


