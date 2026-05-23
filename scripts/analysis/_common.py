"""Shared helpers for printing paper-vs-computed checks.

A CheckResult captures one claim from the paper plus what we get from
re-running the analysis. `render_section` prints them as a small table
with a tick or cross per row.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CheckResult:
    label: str
    paper: str
    computed: str
    passed: bool


def check_close(label: str, paper: float, computed: float, tol: float = 0.01) -> CheckResult:
    """Pass if |paper - computed| < tol. Formats both as +.3f."""
    passed = abs(paper - computed) < tol
    return CheckResult(label, f"{paper:+.3f}", f"{computed:+.3f}", passed)


def check_pct(label: str, paper: float, computed: float, tol: float = 0.01) -> CheckResult:
    """Pass if |paper - computed| < tol. Formats both as percentages."""
    passed = abs(paper - computed) < tol
    return CheckResult(label, f"{paper * 100:.1f}%", f"{computed * 100:.1f}%", passed)


def check_count(label: str, paper: int, computed: int, of: int | None = None) -> CheckResult:
    """Pass if paper == computed. Optional /of suffix for n / total."""
    passed = paper == computed
    if of is not None:
        return CheckResult(label, f"{paper}/{of}", f"{computed}/{of}", passed)
    return CheckResult(label, str(paper), str(computed), passed)


def check_within(label: str, low: float, high: float, computed: float) -> CheckResult:
    """Pass if low <= computed <= high. Use for paper claims like 'within ±0.2'."""
    passed = low <= computed <= high
    return CheckResult(label, f"[{low:+.2f}, {high:+.2f}]", f"{computed:+.3f}", passed)


def render_section(section: str, results: list[CheckResult]) -> int:
    """Print results as a small fixed-width table. Returns the failure count."""
    print(f"\n[{section}]")
    label_w = max(len(r.label) for r in results) if results else 30
    paper_w = max((len(r.paper) for r in results), default=8)
    comp_w  = max((len(r.computed) for r in results), default=8)
    for r in results:
        tick = "✓" if r.passed else "✗"
        print(f"  {r.label:<{label_w}s}  paper: {r.paper:>{paper_w}s}  "
              f"computed: {r.computed:>{comp_w}s}   {tick}")
    n_pass = sum(1 for r in results if r.passed)
    fails  = len(results) - n_pass
    print(f"  → {n_pass}/{len(results)} passing")
    return fails
