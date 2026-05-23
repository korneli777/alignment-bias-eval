"""Verification gate: every number cited in the paper matches the released data.

Each section runner under `scripts/analysis/` recomputes its claims from the
cached parquets and asserts them against the paper's reported values, returning
the number of failed checks. If any check drifts, the matching test fails and
prints which one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))


@pytest.fixture(scope="module")
def section_runners():
    import bbq_decomposition  # noqa: WPS433  (script-style import)
    import chat_template_stats
    import probing_stats
    return {
        "chat_template (§3.1)": chat_template_stats.run,
        "bbq_decomposition (§3.2)": bbq_decomposition.run,
        "probing (§3.3)": probing_stats.run,
    }


def test_all_paper_numbers(section_runners):
    fails: dict[str, int] = {}
    for label, run in section_runners.items():
        n = run()
        if n != 0:
            fails[label] = n
    assert not fails, f"paper-number checks drifted: {fails}"
