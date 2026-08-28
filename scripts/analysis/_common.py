"""Shared formatting for command-line analysis reports."""

from __future__ import annotations


def render_section(section: str, rows: list[tuple[str, str]]) -> None:
    """Print a compact two-column analysis summary."""
    print(f"\n[{section}]")
    label_width = max((len(label) for label, _ in rows), default=30)
    for label, value in rows:
        print(f"  {label:<{label_width}s}  {value}")
