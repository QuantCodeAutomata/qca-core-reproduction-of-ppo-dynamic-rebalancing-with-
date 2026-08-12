"""Tests for src.bootstrap.CircularBlockBootstrap."""

import numpy as np
import pandas as pd
import pytest

from src.bootstrap import CircularBlockBootstrap

_N = 100
_FEAT_COLS = [
    'strat_1', 'strat_2', 'strat_3',
    'hy_spread', 'vix_chg', 'move_proxy',
]
_RET_COLS = ['strat_1', 'strat_2', 'strat_3']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_features(n: int = _N, seed: int = 0) -> pd.DataFrame:
    """Synthetic 6-column features DataFrame."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, len(_FEAT_COLS))) * 0.01
    return pd.DataFrame(data, columns=_FEAT_COLS)


def make_returns(n: int = _N, seed: int = 1) -> pd.DataFrame:
    """Synthetic 3-column strategy returns DataFrame."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, len(_RET_COLS))) * 0.01
    return pd.DataFrame(data, columns=_RET_COLS)


# ---------------------------------------------------------------------------
# Shape and columns
# ---------------------------------------------------------------------------

def test_bootstrap_output_shape():
    """Bootstrapped features and returns DataFrames have the same shape as their inputs."""
    feat_df = make_features()
    ret_df = make_returns()

    cbb = CircularBlockBootstrap(block_length_ratio=0.8, seed=42)
    boot_feat = cbb.bootstrap(feat_df)
    boot_ret = cbb.bootstrap(ret_df)

    assert boot_feat.shape == feat_df.shape, \
        f"Features shape mismatch: {boot_feat.shape} vs {feat_df.shape}"
    assert boot_ret.shape == ret_df.shape, \
        f"Returns shape mismatch: {boot_ret.shape} vs {ret_df.shape}"


def test_bootstrap_preserves_columns():
    """Column names are unchanged after bootstrapping."""
    feat_df = make_features()
    ret_df = make_returns()

    cbb = CircularBlockBootstrap(block_length_ratio=0.8, seed=42)
    boot_feat = cbb.bootstrap(feat_df)
    boot_ret = cbb.bootstrap(ret_df)

    assert list(boot_feat.columns) == list(feat_df.columns), \
        f"Feature columns changed: {list(boot_feat.columns)}"
    assert list(boot_ret.columns) == list(ret_df.columns), \
        f"Return columns changed: {list(boot_ret.columns)}"


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------

def test_bootstrap_returns_valid_values():
    """Bootstrapped output contains no NaN values."""
    feat_df = make_features()
    ret_df = make_returns()

    cbb = CircularBlockBootstrap(block_length_ratio=0.8, seed=42)
    boot_feat = cbb.bootstrap(feat_df)
    boot_ret = cbb.bootstrap(ret_df)

    assert not boot_feat.isna().any().any(), "NaN found in bootstrapped features"
    assert not boot_ret.isna().any().any(), "NaN found in bootstrapped returns"


def test_bootstrap_values_come_from_original():
    """Every row in the bootstrapped output must be an exact row from the original."""
    df = make_features()
    cbb = CircularBlockBootstrap(block_length_ratio=0.8, seed=42)
    bootstrapped = cbb.bootstrap(df)

    original_rows = set(map(tuple, df.values))
    for row in bootstrapped.values:
        assert tuple(row) in original_rows, \
            "Bootstrapped row not found in original data (circular block bootstrap must resample rows)"


def test_bootstrap_different_from_original():
    """Bootstrapped data should not be a row-for-row exact copy of the original."""
    df = make_features()
    cbb = CircularBlockBootstrap(block_length_ratio=0.8, seed=42)
    bootstrapped = cbb.bootstrap(df)

    assert not np.array_equal(df.values, bootstrapped.values), \
        "Bootstrapped data is identical to original — expected a resampled order"


# ---------------------------------------------------------------------------
# Reproducibility and seed behaviour
# ---------------------------------------------------------------------------

def test_different_seeds_give_different_results():
    """Regenerating with different seed offsets produces distinct samples."""
    df = make_returns()
    cbb = CircularBlockBootstrap(block_length_ratio=0.8, seed=42)

    sample_a = cbb.regenerate(df, seed_offset=0)
    sample_b = cbb.regenerate(df, seed_offset=1)

    assert not np.array_equal(sample_a.values, sample_b.values), \
        "Different seed offsets produced identical bootstrap samples"


def test_same_seed_gives_same_results():
    """regenerate with the same seed offset is deterministic."""
    df = make_returns()
    cbb = CircularBlockBootstrap(block_length_ratio=0.8, seed=42)

    sample_a = cbb.regenerate(df, seed_offset=0)
    sample_b = cbb.regenerate(df, seed_offset=0)

    assert np.array_equal(sample_a.values, sample_b.values), \
        "Same seed offset should produce identical samples (reproducibility)"


# ---------------------------------------------------------------------------
# Block length fractions
# ---------------------------------------------------------------------------

def test_block_length_fracs():
    """Block length ratios 0.7, 0.8, 0.9 all produce valid output of identical shape."""
    df = make_features()

    for frac in [0.7, 0.8, 0.9]:
        cbb = CircularBlockBootstrap(block_length_ratio=frac, seed=42)
        result = cbb.bootstrap(df)

        assert result.shape == df.shape, \
            f"Shape mismatch for block_length_ratio={frac}: {result.shape} vs {df.shape}"
        assert not result.isna().any().any(), \
            f"NaN found in output for block_length_ratio={frac}"
        assert list(result.columns) == list(df.columns), \
            f"Columns changed for block_length_ratio={frac}"
