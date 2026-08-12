"""Tests for src.features.FeatureBuilder using synthetic price data.

No file I/O is performed: a synthetic price DataFrame is injected directly
into the FeatureBuilder instance, bypassing the CSV read in __init__.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import FeatureBuilder, OBS_FEATURE_COLS

# Expected column names for the 15-dimensional observation feature vector
_EXPECTED_FEATURE_COLS = [
    'strat_1', 'strat_2', 'strat_3',
    'hy_spread', 'vix_chg', 'move_proxy',
    'mean40_s1', 'mean40_s2', 'mean40_s3',
    'std60_s1', 'std60_s2', 'std60_s3',
    'std60_ctx1', 'std60_ctx2', 'std60_ctx3',
]

# Expected strategy returns columns
_STRAT_COLS = ['strat_1', 'strat_2', 'strat_3']

# Rolling warm-up: pct_change drops 1 row; rolling(60).std drops 59 more → 60 total
_WARMUP = 60

# Blend weights matching _W2 in features.py
_W2 = {'dev_eq': 0.55, 'em_eq': 0.05, 'credit': 0.20, 'govies': 0.20}


# ---------------------------------------------------------------------------
# Helper: build a FeatureBuilder with synthetic prices (no file IO)
# ---------------------------------------------------------------------------

def _make_feature_builder(n: int = 200, seed: int = 42) -> FeatureBuilder:
    """Return a FeatureBuilder whose _prices is set to synthetic data.

    Required price columns: EFA, EEM, LQD, TLT, HYG, ^VIX, ^TNX.
    All series start at 100.0 and evolve as a random walk.
    """
    rng = np.random.default_rng(seed)
    # Small random daily moves; keep prices positive (start at 100)
    daily_moves = rng.standard_normal((n, 7)) * 0.5
    prices = 100.0 + np.cumsum(daily_moves, axis=0)
    prices = np.abs(prices) + 1.0   # ensure strictly positive

    price_cols = ['EFA', 'EEM', 'LQD', 'TLT', 'HYG', '^VIX', '^TNX']
    price_df = pd.DataFrame(prices, columns=price_cols)

    fb = object.__new__(FeatureBuilder)
    fb._prices = price_df
    return fb


# ---------------------------------------------------------------------------
# Column structure tests
# ---------------------------------------------------------------------------

def test_output_columns():
    """features_df has exactly 15 columns matching OBS_FEATURE_COLS."""
    fb = _make_feature_builder()
    feat_df = fb.build()

    assert list(feat_df.columns) == _EXPECTED_FEATURE_COLS, \
        f"Unexpected columns: {list(feat_df.columns)}"
    assert len(feat_df.columns) == 15, \
        f"Expected 15 columns, got {len(feat_df.columns)}"


def test_obs_feature_cols_constant_matches():
    """OBS_FEATURE_COLS module constant equals the expected list."""
    assert OBS_FEATURE_COLS == _EXPECTED_FEATURE_COLS, \
        f"OBS_FEATURE_COLS mismatch: {OBS_FEATURE_COLS}"


def test_strategy_returns_columns():
    """get_strategy_returns() returns a DataFrame with columns [strat_1, strat_2, strat_3]."""
    fb = _make_feature_builder()
    ret_df = fb.get_strategy_returns()

    assert list(ret_df.columns) == _STRAT_COLS, \
        f"Strategy returns columns: {list(ret_df.columns)}"
    assert len(ret_df.columns) == 3


def test_observation_dim():
    """The base feature DataFrame has exactly 15 columns (obs vector before runtime appends)."""
    fb = _make_feature_builder()
    feat_df = fb.get_obs_features()

    assert feat_df.shape[1] == 15, \
        f"Expected 15-dim obs feature vector, got {feat_df.shape[1]}"


# ---------------------------------------------------------------------------
# No-NaN guarantee beyond warm-up
# ---------------------------------------------------------------------------

def test_no_nan_after_warmup():
    """build() drops all NaN rows; the returned DataFrame contains no NaN values."""
    fb = _make_feature_builder(n=200)
    feat_df = fb.build()

    has_nan = feat_df.isna().any().any()
    assert not has_nan, \
        f"NaN found in features_df after build():\n{feat_df.isna().sum()}"


def test_row_count_after_warmup():
    """After dropping the 60-row warm-up, the returned DataFrame has n-60 rows."""
    n = 200
    fb = _make_feature_builder(n=n)
    feat_df = fb.build()

    expected_rows = n - _WARMUP
    assert len(feat_df) == expected_rows, \
        f"Expected {expected_rows} rows after warm-up, got {len(feat_df)}"


# ---------------------------------------------------------------------------
# Strategy weight formula
# ---------------------------------------------------------------------------

def test_strategy_weights_strat1_equals_efa():
    """strat_1 in features_df equals EFA daily pct_change (dev_eq sleeve)."""
    fb = _make_feature_builder(n=200)
    prices = fb._prices
    feat_df = fb.build()

    efa_ret = prices['EFA'].pct_change()
    expected = efa_ret.loc[feat_df.index].values

    np.testing.assert_allclose(
        feat_df['strat_1'].values, expected, rtol=1e-10,
        err_msg="strat_1 should equal EFA daily pct_change"
    )


def test_strategy_weights_strat3_equals_tlt():
    """strat_3 in features_df equals TLT daily pct_change (govies sleeve)."""
    fb = _make_feature_builder(n=200)
    prices = fb._prices
    feat_df = fb.build()

    tlt_ret = prices['TLT'].pct_change()
    expected = tlt_ret.loc[feat_df.index].values

    np.testing.assert_allclose(
        feat_df['strat_3'].values, expected, rtol=1e-10,
        err_msg="strat_3 should equal TLT daily pct_change"
    )


def test_strategy_weights_strat2_blend():
    """strat_2 ≈ 0.55*EFA + 0.05*EEM + 0.20*LQD + 0.20*TLT daily returns."""
    fb = _make_feature_builder(n=200)
    prices = fb._prices
    feat_df = fb.build()

    idx = feat_df.index
    efa_ret = prices['EFA'].pct_change().loc[idx].values
    eem_ret = prices['EEM'].pct_change().loc[idx].values
    lqd_ret = prices['LQD'].pct_change().loc[idx].values
    tlt_ret = prices['TLT'].pct_change().loc[idx].values

    expected_strat2 = (
        _W2['dev_eq'] * efa_ret
        + _W2['em_eq'] * eem_ret
        + _W2['credit'] * lqd_ret
        + _W2['govies'] * tlt_ret
    )

    np.testing.assert_allclose(
        feat_df['strat_2'].values, expected_strat2, rtol=1e-10,
        err_msg="strat_2 should equal 0.55*EFA + 0.05*EEM + 0.20*LQD + 0.20*TLT returns"
    )


# ---------------------------------------------------------------------------
# Date-range filtering (uses DatetimeIndex)
# ---------------------------------------------------------------------------

def test_date_range_filter():
    """Passing start_date/end_date to build() restricts the returned rows."""
    n = 300
    fb = _make_feature_builder(n=n)
    # Give _prices a DatetimeIndex so the date filter can be applied
    dates = pd.date_range('2000-01-01', periods=n, freq='B')
    fb._prices.index = dates

    feat_all = fb.build()
    start = '2001-01-01'
    feat_filtered = fb.build(start_date=start)

    # Filtered result should be a subset of the full result
    assert len(feat_filtered) <= len(feat_all), \
        "Filtered features should have fewer or equal rows than unfiltered"
    if len(feat_filtered) > 0:
        assert feat_filtered.index.min() >= pd.Timestamp(start), \
            "Filtered result includes dates before start_date"
