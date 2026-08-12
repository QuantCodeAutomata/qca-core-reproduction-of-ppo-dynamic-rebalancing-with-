import numpy as np
import pandas as pd
import pytest
from src.environment import PortfolioEnv
from src.oracle import OracleOptimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    'strat_1', 'strat_2', 'strat_3',
    'hy_spread', 'vix_chg', 'move_proxy',
    'mean40_s1', 'mean40_s2', 'mean40_s3',
    'std60_s1', 'std60_s2', 'std60_s3',
    'std60_ctx1', 'std60_ctx2', 'std60_ctx3',
]

RETURN_COLS = ['strat_1', 'strat_2', 'strat_3']


def make_features(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, len(FEATURE_COLS))) * 0.01
    return pd.DataFrame(data, columns=FEATURE_COLS)


def make_returns(n: int = 100, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, 3)) * 0.01
    return pd.DataFrame(data, columns=RETURN_COLS)


def make_env(n: int = 100, reward_type: str = 'return_only', **kwargs) -> PortfolioEnv:
    return PortfolioEnv(
        features_df=make_features(n),
        strategy_returns_df=make_returns(n),
        reward_type=reward_type,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_observation_space_dim():
    env = make_env()
    obs, _ = env.reset()
    assert obs.shape == (19,), f'Expected (19,), got {obs.shape}'


def test_action_softmax():
    env = make_env()
    env.reset()
    action = np.array([1.5, -2.0, 0.5], dtype=np.float32)
    _, _, _, _, info = env.step(action)
    weights = info['weights']
    assert np.isclose(weights.sum(), 1.0, atol=1e-6), \
        f'Weights do not sum to 1: {weights.sum()}'


def test_weights_nonnegative():
    env = make_env()
    env.reset()
    # Step several times with varied actions, verify all weights >= 0
    for _ in range(10):
        action = env.action_space.sample()
        _, _, terminated, _, info = env.step(action)
        assert np.all(info['weights'] >= 0.0), \
            f'Negative weights found: {info["weights"]}'
        if terminated:
            break


def test_episode_terminates():
    T = 20
    env = make_env(n=T)
    env.reset()
    steps = 0
    last_terminated = False
    while True:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
        if terminated or truncated:
            last_terminated = terminated
            break
    # Episode ends after T-1 steps (t goes 0..T-2, terminates when t==T-1)
    assert steps == T - 1, f'Expected {T - 1} steps, got {steps}'
    assert last_terminated


def test_reward_type_regret():
    oracle = OracleOptimizer(n=5)
    env = make_env(n=50, reward_type='regret', oracle=oracle)
    env.reset()
    # Should run without error
    _, reward, _, _, _ = env.step(np.zeros(3, dtype=np.float32))
    assert isinstance(reward, float)


def test_reward_type_diff_sharpe():
    env = make_env(n=50, reward_type='diff_sharpe')
    env.reset()
    _, reward, _, _, _ = env.step(np.zeros(3, dtype=np.float32))
    assert isinstance(reward, float)
    # Step a few more times; subsequent rewards may be non-zero
    for _ in range(5):
        _, reward, terminated, _, _ = env.step(env.action_space.sample())
        assert isinstance(reward, float)
        if terminated:
            break


def test_reward_type_embedded_drawdown():
    env = make_env(n=50, reward_type='embedded_drawdown')
    env.reset()
    _, reward, _, _, _ = env.step(np.zeros(3, dtype=np.float32))
    assert isinstance(reward, float)
    for _ in range(5):
        _, reward, terminated, _, _ = env.step(env.action_space.sample())
        assert isinstance(reward, float)
        if terminated:
            break


def test_reward_type_return_only():
    # With tc_schedule=None (tc_train=0) and zero turnover (equal weights),
    # reward (gross return) equals net_ret (also gross return when turnover=0).
    env = make_env(n=50, reward_type='return_only', tc_schedule=None)
    env.reset()
    # action=[0,0,0] -> softmax -> [1/3,1/3,1/3] = prev_weights -> turnover=0
    action = np.zeros(3, dtype=np.float32)
    _, reward, _, _, info = env.step(action)
    # turnover=0 => net_ret = gross_ret; tc_train=0 => reward = gross_ret
    assert np.isclose(reward, info['net_ret'], atol=1e-7), \
        f'reward {reward} != net_ret {info["net_ret"]}'


def test_reset_clears_state():
    env = make_env(n=50, reward_type='diff_sharpe')
    env.reset()

    # Step several times to accumulate state in A_t, B_t, nav
    for _ in range(10):
        _, _, terminated, _, _ = env.step(env.action_space.sample())
        if terminated:
            break

    # Confirm state has drifted from initial values
    assert env.nav != 1.0 or env.A_t != 0.0 or env.B_t != 0.0

    # Reset and verify all path-dependent state is cleared
    env.reset()
    assert env.A_t == 0.0, f'A_t not reset: {env.A_t}'
    assert env.B_t == 0.0, f'B_t not reset: {env.B_t}'
    assert env.nav == 1.0, f'nav not reset: {env.nav}'
    assert env.peak_nav == 1.0, f'peak_nav not reset: {env.peak_nav}'
    assert env.mdd_running == 0.0, f'mdd_running not reset: {env.mdd_running}'
    assert env.t == 0, f't not reset: {env.t}'
    assert np.allclose(env.prev_weights, [1/3, 1/3, 1/3])
