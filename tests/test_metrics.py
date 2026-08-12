"""Tests for src.metrics performance metric functions."""

import math
import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    compute_annual_return,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_calmar_ratio,
    compute_omega_ratio,
    compute_sortino_ratio,
)

_ALL_METRICS = [
    compute_annual_return,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_calmar_ratio,
    compute_omega_ratio,
    compute_sortino_ratio,
]


# ---------------------------------------------------------------------------
# Annual return
# ---------------------------------------------------------------------------

def test_annual_return_positive():
    """Known positive daily returns should produce a positive CAGR."""
    returns = pd.Series([0.001] * 252)
    result = compute_annual_return(returns)
    assert result > 0, f"Expected positive CAGR, got {result}"


def test_annual_return_negative():
    """Known negative daily returns should produce a negative CAGR."""
    returns = pd.Series([-0.001] * 252)
    result = compute_annual_return(returns)
    assert result < 0, f"Expected negative CAGR, got {result}"


def test_annual_return_known_value():
    """One year of 0.001 daily returns → CAGR ≈ 1.001^252 − 1 ≈ 0.2838."""
    returns = pd.Series([0.001] * 252)
    expected = 1.001 ** 252 - 1.0
    result = compute_annual_return(returns)
    assert math.isclose(result, expected, rel_tol=1e-9), \
        f"Expected CAGR ≈ {expected:.6f}, got {result:.6f}"


# ---------------------------------------------------------------------------
# Max drawdown
# ---------------------------------------------------------------------------

def test_max_drawdown_negative():
    """Max drawdown is non-positive for any return series with a loss period."""
    returns = pd.Series([0.01, 0.01, -0.05, 0.02, 0.01, -0.03, 0.02])
    result = compute_max_drawdown(returns)
    assert result <= 0, f"Max drawdown should be <= 0, got {result}"


def test_max_drawdown_zero_for_monotone_positive():
    """A strictly increasing NAV (all positive returns) has zero max drawdown."""
    returns = pd.Series([0.001] * 100)
    result = compute_max_drawdown(returns)
    assert result == 0.0, f"Monotone positive returns should have MDD=0, got {result}"


def test_max_drawdown_known_value():
    """Single loss of 10 % from a peak of 1.0 gives MDD = −0.10."""
    # NAV: 1.0 → 1.0 (at t=0 before any step)
    # day 1: +0 % → peak=1.0; day 2: -10 % → NAV=0.9, drawdown=-0.10
    returns = pd.Series([0.0, -0.10])
    result = compute_max_drawdown(returns)
    assert math.isclose(result, -0.10, rel_tol=1e-9), \
        f"Expected MDD=-0.10, got {result}"


# ---------------------------------------------------------------------------
# Sharpe ratio
# ---------------------------------------------------------------------------

def test_sharpe_zero_mean():
    """Returns with exactly zero mean should yield a Sharpe ratio of 0."""
    returns = pd.Series([0.01, -0.01] * 50)
    result = compute_sharpe_ratio(returns)
    assert math.isclose(result, 0.0, abs_tol=1e-10), \
        f"Expected Sharpe ≈ 0 for zero-mean returns, got {result}"


def test_sharpe_positive_for_positive_mean():
    """Consistently positive returns should give a positive Sharpe ratio."""
    returns = pd.Series([0.001] * 252)
    result = compute_sharpe_ratio(returns)
    # std of constant series is 0 → function returns 0.0 per contract
    # Use noisy positive returns instead
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(loc=0.002, scale=0.005, size=252))
    result = compute_sharpe_ratio(returns)
    assert result > 0, f"Expected positive Sharpe for positive-mean returns, got {result}"


# ---------------------------------------------------------------------------
# Calmar ratio
# ---------------------------------------------------------------------------

def test_calmar_ratio_positive_for_positive_returns():
    """Mostly positive returns with a small drawdown should yield positive Calmar."""
    returns = pd.Series([0.002] * 60 + [-0.005] * 5 + [0.002] * 60)
    calmar = compute_calmar_ratio(returns)
    assert calmar > 0, f"Expected positive Calmar ratio, got {calmar}"


def test_calmar_ratio_zero_when_no_drawdown():
    """When MDD is zero (all positive returns), Calmar ratio is defined as 0.0."""
    returns = pd.Series([0.001] * 100)
    result = compute_calmar_ratio(returns)
    # MDD=0 → function returns 0.0 per contract
    assert result == 0.0, f"Expected Calmar=0.0 when MDD=0, got {result}"


# ---------------------------------------------------------------------------
# Omega ratio
# ---------------------------------------------------------------------------

def test_omega_above_one_for_positive_returns():
    """Mostly positive returns yield Omega > 1."""
    returns = pd.Series([0.005] * 90 + [-0.002] * 10)
    result = compute_omega_ratio(returns)
    assert result > 1.0, f"Expected Omega > 1 for mostly-positive returns, got {result}"


def test_omega_below_one_for_negative_returns():
    """Mostly negative returns yield Omega < 1."""
    returns = pd.Series([-0.005] * 90 + [0.002] * 10)
    result = compute_omega_ratio(returns)
    assert result < 1.0, f"Expected Omega < 1 for mostly-negative returns, got {result}"


def test_omega_inf_when_no_losses():
    """Returns all above threshold → Omega is infinite."""
    returns = pd.Series([0.01] * 50)
    result = compute_omega_ratio(returns, threshold=0.0)
    assert result == np.inf, f"Expected Omega=inf when no losses, got {result}"


# ---------------------------------------------------------------------------
# Edge cases: single-element and empty series
# ---------------------------------------------------------------------------

def test_single_return():
    """All metric functions handle a single-element series without raising."""
    s = pd.Series([0.01])
    for fn in _ALL_METRICS:
        result = fn(s)
        # Result must be a finite float or inf (no crash, no NaN for these functions)
        assert isinstance(result, float), \
            f"{fn.__name__} returned non-float for single element: {result}"


def test_empty_series():
    """All metric functions handle an empty series and return 0.0 without raising."""
    s = pd.Series([], dtype=float)
    for fn in _ALL_METRICS:
        result = fn(s)
        assert isinstance(result, float), \
            f"{fn.__name__} returned non-float for empty series: {result}"
        assert result == 0.0 or math.isnan(result), \
            f"{fn.__name__} returned unexpected value {result} for empty series"
