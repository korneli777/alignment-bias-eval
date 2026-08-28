"""Unit tests for biaseval.analysis.statistics on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from biaseval.analysis.statistics import (
    bootstrap_ci,
    bootstrap_paired_delta_ci,
    cohens_d_paired,
    holm_bonferroni,
    paired_binary_effect_summary,
    paired_permutation_test,
)

# ─── Cohen's d (paired) ──────────────────────────────────────────────────────


def test_cohens_d_paired_zero_difference():
    """Identical arrays → d = 0 / 0 → NaN (no signal, no variance)."""
    x = np.array([0.5, 0.5, 0.5, 0.5])
    assert np.isnan(cohens_d_paired(x, x))


def test_cohens_d_paired_constant_offset():
    """diff = +1 everywhere → std(diff) = 0 → NaN (no variability)."""
    base = np.zeros(10)
    inst = np.ones(10)
    assert np.isnan(cohens_d_paired(base, inst))


def test_cohens_d_paired_known_value():
    """diff = [0, 1, 2, 3]: mean = 1.5, sd_ddof1 = sqrt(5/3*1) ≈ 1.291
    → d ≈ 1.162. Sanity-checks the formula on a small example.
    """
    base = np.array([0.0, 0.0, 0.0, 0.0])
    inst = np.array([0.0, 1.0, 2.0, 3.0])
    d = cohens_d_paired(base, inst)
    assert d == pytest.approx(1.162, abs=0.005)


def test_cohens_d_paired_size_mismatch_returns_nan():
    assert np.isnan(cohens_d_paired(np.zeros(5), np.zeros(6)))


# ─── Paired permutation test ─────────────────────────────────────────────────


def test_paired_permutation_identical_arrays():
    """Identical paired samples → no extreme deltas → p = 1."""
    x = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0], dtype=float)
    out = paired_permutation_test(x, x, n_perm=500)
    assert out["observed_delta"] == 0.0
    assert out["p_value"] == 1.0


def test_paired_permutation_clearly_different():
    """Strong consistent signal → p very small."""
    rng = np.random.default_rng(0)
    base = rng.binomial(1, 0.2, size=200).astype(float)
    inst = rng.binomial(1, 0.8, size=200).astype(float)
    out = paired_permutation_test(base, inst, n_perm=2000)
    assert out["observed_delta"] > 0.4
    assert out["p_value"] < 0.01


# ─── Holm-Bonferroni ────────────────────────────────────────────────────────


def test_holm_smallest_p_gets_largest_correction():
    """Holm multiplies the smallest p by m, the next by m-1, etc."""
    pvals = {"a": 0.001, "b": 0.04, "c": 0.30}
    out = holm_bonferroni(pvals, alpha=0.05)
    # a: 0.001 * 3 = 0.003 (reject)
    assert out["a"]["p_adj"] == pytest.approx(0.003, abs=1e-9)
    assert out["a"]["reject"]
    # b: 0.04 * 2 = 0.08 (do not reject — note monotonicity keeps it ≥ a's adj)
    assert out["b"]["p_adj"] == pytest.approx(0.08, abs=1e-9)
    assert not out["b"]["reject"]
    # c: 0.30 * 1 = 0.30 (do not reject), monotonic ≥ b
    assert out["c"]["p_adj"] == pytest.approx(0.30, abs=1e-9)


def test_holm_enforces_monotonicity():
    """Adjusted p must be monotone in the original p ordering."""
    # If raw p × (m - i) violates monotonicity, Holm clamps to the running max.
    pvals = {"a": 0.04, "b": 0.04, "c": 0.04}  # m=3 → 0.12, 0.08, 0.04
    out = holm_bonferroni(pvals, alpha=0.05)
    adjs = sorted([out[k]["p_adj"] for k in "abc"])
    # All three should land at 0.12 due to running-max enforcement.
    assert all(a == pytest.approx(0.12, abs=1e-9) for a in adjs)


def test_holm_empty_input():
    assert holm_bonferroni({}) == {}


def test_holm_drops_nan_p_values():
    out = holm_bonferroni({"a": 0.01, "b": float("nan"), "c": 0.05})
    assert "b" not in out


# ─── Bootstrap CI ────────────────────────────────────────────────────────────


def test_bootstrap_ci_point_inside_bounds():
    rng = np.random.default_rng(1)
    values = rng.normal(loc=2.0, scale=1.0, size=300)
    point, lo, hi = bootstrap_ci(values, n_iter=1000, rng=rng)
    assert lo <= point <= hi
    assert point == pytest.approx(values.mean(), abs=1e-9)


def test_bootstrap_paired_delta_point_matches_observed():
    rng = np.random.default_rng(2)
    base = rng.normal(0.5, 0.1, size=100)
    inst = rng.normal(0.3, 0.1, size=100)
    point, lo, hi = bootstrap_paired_delta_ci(base, inst, n_iter=1000, seed=2)
    assert lo <= point <= hi
    # Point estimate is the observed mean diff.
    assert point == pytest.approx(inst.mean() - base.mean(), abs=1e-9)


def test_bootstrap_paired_delta_size_mismatch_returns_nan():
    point, lo, hi = bootstrap_paired_delta_ci(np.zeros(5), np.zeros(6))
    assert np.isnan(point) and np.isnan(lo) and np.isnan(hi)


def test_paired_binary_effect_summary_uses_item_level_differences():
    result = paired_binary_effect_summary(
        [True, True, True, False],
        [False, True, False, True],
    )

    assert result["n_items"] == 4
    assert result["n_base_positive"] == 3
    assert result["n_instruct_positive"] == 2
    assert result["n_decrease"] == 2
    assert result["n_increase"] == 1
    assert result["n_unchanged"] == 1
    assert result["mean_difference"] == -0.25
    assert result["sd_difference"] == pytest.approx(0.957427, abs=1e-6)
    assert result["cohens_d"] == pytest.approx(-0.261116, abs=1e-6)
