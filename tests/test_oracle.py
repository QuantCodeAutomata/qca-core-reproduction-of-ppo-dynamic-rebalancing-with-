"""Tests for src.oracle.OracleOptimizer."""

import numpy as np
import pandas as pd
import pytest

from src.oracle import OracleOptimizer

_COLS = ['strat_1', 'strat_2', 'strat_3']
_PREV_W = np.array([1 / 3, 1 / 3, 1 / 3])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_returns(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Random normal strategy returns with columns strat_1/2/3."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, 3)) * 0.01
    return pd.DataFrame(data, columns=_COLS)


# ---------------------------------------------------------------------------
# Oracle weight properties
# ---------------------------------------------------------------------------

def test_oracle_weights_sum_to_one():
    """Oracle weights must sum exactly to 1 (simplex constraint)."""
    oracle = OracleOptimizer(n=14)
    rets = make_returns(n=60)
    w = oracle.compute_oracle_weights(
        rets, t=20, prev_weights=_PREV_W, tc_train=0.0, split_end_idx=59
    )
    assert np.isclose(w.sum(), 1.0, atol=1e-6), \
        f"Oracle weights sum to {w.sum()}, expected 1.0"


def test_oracle_weights_nonnegative():
    """Oracle weights are non-negative (long-only constraint)."""
    oracle = OracleOptimizer(n=14)
    rets = make_returns(n=60)
    w = oracle.compute_oracle_weights(
        rets, t=20, prev_weights=_PREV_W, tc_train=0.0, split_end_idx=59
    )
    assert np.all(w >= -1e-10), \
        f"Oracle weights have negative values: {w}"


def test_oracle_weights_shape():
    """Oracle returns a weight vector of length 3."""
    oracle = OracleOptimizer(n=14)
    rets = make_returns(n=60)
    w = oracle.compute_oracle_weights(
        rets, t=20, prev_weights=_PREV_W, tc_train=0.0, split_end_idx=59
    )
    assert w.shape == (3,), f"Expected shape (3,), got {w.shape}"


def test_oracle_single_best_strategy():
    """When one strategy has much higher returns, oracle should weight it the most."""
    n = 60
    data = np.zeros((n, 3))
    data[:, 0] = 0.02   # strat_1 dominates
    data[:, 1] = 0.001
    data[:, 2] = 0.001
    rets = pd.DataFrame(data, columns=_COLS)

    oracle = OracleOptimizer(n=14)
    w = oracle.compute_oracle_weights(
        rets, t=10, prev_weights=_PREV_W, tc_train=0.0, split_end_idx=59
    )
    # strat_1 has the highest expected return → should receive the most weight
    assert w[0] > w[1], f"strat_1 weight {w[0]:.4f} should exceed strat_2 {w[1]:.4f}"
    assert w[0] > w[2], f"strat_1 weight {w[0]:.4f} should exceed strat_3 {w[2]:.4f}"


def test_oracle_fallback_equal_weights_at_boundary():
    """When t >= split_end_idx the oracle cannot look forward and returns equal weights."""
    oracle = OracleOptimizer(n=14)
    rets = make_returns(n=30)
    # t_fwd = min(29+14, 29, 29) = 29 which equals t=29; triggers fallback
    w = oracle.compute_oracle_weights(
        rets, t=29, prev_weights=_PREV_W, tc_train=0.0, split_end_idx=29
    )
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert np.all(w >= -1e-10)


# ---------------------------------------------------------------------------
# Regret reward
# ---------------------------------------------------------------------------

def test_regret_reward_is_float():
    """Regret reward must be a Python float."""
    oracle = OracleOptimizer(n=14)
    rets = make_returns(n=60)
    w_t = np.array([1 / 3, 1 / 3, 1 / 3])
    w_star = np.array([0.5, 0.3, 0.2])
    reward = oracle.compute_regret_reward(rets, t=10, w_t=w_t, w_star=w_star)
    assert isinstance(reward, float), f"Expected float, got {type(reward)}"


def test_regret_reward_negative_when_oracle_outperforms():
    """Regret reward is negative when the oracle's weights far outperform the agent's."""
    n = 60
    # strat_1 has much higher return than strat_3
    data = np.zeros((n, 3))
    data[:, 0] = 0.05   # strat_1 high
    data[:, 1] = 0.005
    data[:, 2] = 0.005  # strat_3 low
    rets = pd.DataFrame(data, columns=_COLS)

    oracle = OracleOptimizer(n=14)
    # Agent bets entirely on strat_3, oracle bets entirely on strat_1
    w_t = np.array([0.0, 0.0, 1.0])
    w_star = np.array([1.0, 0.0, 0.0])
    reward = oracle.compute_regret_reward(rets, t=5, w_t=w_t, w_star=w_star)
    # reward = -rho_bar @ (w_star - w_t)
    # rho_bar ≈ [0.05, 0.005, 0.005]
    # w_star - w_t = [1, 0, -1]
    # rho_bar @ diff = 0.05 - 0.005 = 0.045 > 0  →  reward = -0.045 < 0
    assert reward < 0.0, \
        f"Expected negative regret reward when oracle outperforms, got {reward}"


def test_regret_reward_zero_when_equal_weights():
    """Regret reward is 0 when agent and oracle choose the same weights."""
    oracle = OracleOptimizer(n=14)
    rets = make_returns(n=60)
    w = np.array([0.5, 0.3, 0.2])
    reward = oracle.compute_regret_reward(rets, t=10, w_t=w, w_star=w.copy())
    assert reward == 0.0, \
        f"Expected reward=0 when w_t == w_star, got {reward}"


# ---------------------------------------------------------------------------
# split_end_idx enforcement
# ---------------------------------------------------------------------------

def test_oracle_respects_split_end_idx():
    """Oracle weights remain valid when split_end_idx constrains the look-ahead window."""
    n = 100
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n, 3)) * 0.01
    rets = pd.DataFrame(data, columns=_COLS)

    oracle = OracleOptimizer(n=14)
    # Only allow oracle to look up to index 20
    w = oracle.compute_oracle_weights(
        rets, t=5, prev_weights=_PREV_W, tc_train=0.0, split_end_idx=20
    )
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert np.all(w >= -1e-10)
    assert w.shape == (3,)


def test_oracle_split_end_idx_less_than_n_horizon():
    """split_end_idx smaller than t+n clips the forward window correctly."""
    n = 50
    rng = np.random.default_rng(1)
    data = rng.standard_normal((n, 3)) * 0.01
    rets = pd.DataFrame(data, columns=_COLS)

    oracle = OracleOptimizer(n=14)
    # split_end_idx=12: t_fwd = min(5+14, 12, 49) = 12; forward window is [5:12]
    w = oracle.compute_oracle_weights(
        rets, t=5, prev_weights=_PREV_W, tc_train=0.0, split_end_idx=12
    )
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert np.all(w >= -1e-10)


# ---------------------------------------------------------------------------
# Precompute lookup-table tests
# ---------------------------------------------------------------------------

def test_precompute_returns_self():
    """precompute() returns the OracleOptimizer instance for chaining."""
    oracle = OracleOptimizer(n=14)
    rets = make_returns(n=60)
    result = oracle.precompute(rets, split_end_idx=59)
    assert result is oracle


def test_precompute_rho_table_shape():
    """After precompute, _rho_table has shape (T, 3)."""
    oracle = OracleOptimizer(n=14)
    rets = make_returns(n=60)
    oracle.precompute(rets, split_end_idx=59)
    assert oracle._rho_table is not None
    assert oracle._rho_table.shape == (60, 3)


def test_precompute_sigma_table_shape():
    """After precompute, _sigma_table has shape (T, 3, 3)."""
    oracle = OracleOptimizer(n=14)
    rets = make_returns(n=60)
    oracle.precompute(rets, split_end_idx=59)
    assert oracle._sigma_table is not None
    assert oracle._sigma_table.shape == (60, 3, 3)


def test_precompute_consistent_with_slow_path():
    """Fast precompute path and slow per-step path return consistent oracle weights."""
    np.random.seed(7)
    rets = make_returns(n=80, seed=7)
    split_end = 75

    # Fast path (precompute)
    oracle_fast = OracleOptimizer(n=14, n_grid=50)
    oracle_fast.precompute(rets, split_end)
    w_fast = oracle_fast.compute_oracle_weights(
        rets, t=30, prev_weights=_PREV_W, tc_train=0.001, split_end_idx=split_end
    )

    # Slow path (no precompute — uses _compute_rho_sigma fallback by passing dummy table)
    oracle_slow = OracleOptimizer(n=14, n_grid=50)   # tables = None → fallback
    w_slow = oracle_slow.compute_oracle_weights(
        rets, t=30, prev_weights=_PREV_W, tc_train=0.001, split_end_idx=split_end
    )

    # Both paths use the same grid, so weights must be identical
    np.testing.assert_allclose(w_fast, w_slow, atol=1e-10,
                               err_msg='Fast and slow oracle paths diverge')


def test_precompute_weights_still_on_simplex():
    """compute_oracle_weights with precomputed tables still returns valid simplex point."""
    oracle = OracleOptimizer(n=14)
    rets = make_returns(n=80)
    oracle.precompute(rets, split_end_idx=79)
    for t in [0, 10, 40, 79]:
        w = oracle.compute_oracle_weights(
            rets, t=t, prev_weights=_PREV_W, tc_train=0.0025, split_end_idx=79
        )
        assert np.isclose(w.sum(), 1.0, atol=1e-6), f't={t}: weights sum={w.sum()}'
        assert np.all(w >= -1e-10), f't={t}: negative weights {w}'


def test_simplex_grid_cache():
    """The class-level simplex grid is cached across OracleOptimizer instances."""
    OracleOptimizer._grid_cache.clear()
    oracle1 = OracleOptimizer(n=14, n_grid=30)
    oracle2 = OracleOptimizer(n=14, n_grid=30)
    assert oracle1._W is oracle2._W, 'Grid should be the same cached array'
    assert 30 in OracleOptimizer._grid_cache
