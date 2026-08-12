import numpy as np
import pandas as pd
import pytest
import tempfile
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment import PortfolioEnv
from src.scheduler import TCScheduler
from src.trainer import EntropyDecayCallback, PhaseTrainer


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


def make_features(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, len(FEATURE_COLS))) * 0.01
    return pd.DataFrame(data, columns=FEATURE_COLS)


def make_returns(n: int = 200, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, 3)) * 0.01
    return pd.DataFrame(data, columns=RETURN_COLS)


def make_env(n: int = 200) -> PortfolioEnv:
    return PortfolioEnv(
        features_df=make_features(n),
        strategy_returns_df=make_returns(n),
        reward_type='return_only',
        oracle=None,
        tc_schedule=None,
        split_end_idx=n - 1,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ppo_builds_correctly():
    env = make_env(n=200)
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = PhaseTrainer(
            features_df=make_features(200),
            strategy_returns_df=make_returns(200),
            reward_type='return_only',
            use_bootstrap=False,
            use_tc_schedule=False,
            checkpoint_dir=tmpdir,
        )
        model = trainer._build_ppo(env)

    # Observation and action space shapes
    assert model.observation_space.shape == (19,)
    assert model.action_space.shape == (3,)

    # Policy network: first layer must be Linear(19, 64)
    policy_net = model.policy.mlp_extractor.policy_net
    first_layer = policy_net[0]
    assert isinstance(first_layer, torch.nn.Linear), \
        f'Expected Linear, got {type(first_layer)}'
    assert first_layer.in_features == 19, \
        f'Expected in_features=19, got {first_layer.in_features}'
    assert first_layer.out_features == 64, \
        f'Expected out_features=64, got {first_layer.out_features}'

    # Action head: Linear(64, 3)
    action_net = model.policy.action_net
    assert isinstance(action_net, torch.nn.Linear)
    assert action_net.out_features == 3, \
        f'Expected action out_features=3, got {action_net.out_features}'


def test_entropy_decay_callback():
    env = make_env(n=200)
    vec_env = DummyVecEnv([lambda: env])

    # Use a small n_steps so training completes within 500 timesteps
    model = PPO(
        policy='MlpPolicy',
        env=vec_env,
        n_steps=64,
        batch_size=16,
        ent_coef=5e-5,
        verbose=0,
        seed=42,
    )

    total_timesteps = 500
    cb = EntropyDecayCallback(
        start_val=5e-5,
        decay_fraction=0.5,   # decay over first 250 steps
        total_timesteps=total_timesteps,
    )

    initial_ent = model.ent_coef
    model.learn(total_timesteps=total_timesteps, callback=cb)

    # After 500 steps (> decay_steps=250), entropy coef must be 0
    assert model.ent_coef < initial_ent, \
        f'ent_coef did not decrease: {model.ent_coef} vs initial {initial_ent}'
    assert model.ent_coef == 0.0, \
        f'ent_coef should be 0.0 after decay, got {model.ent_coef}'


def test_tc_scheduler_monotone():
    # episode_length=5 -> S = 500, so 600 steps covers the full ramp and plateau
    sched = TCScheduler(episode_length=5, phase=1)
    tcs = []
    for _ in range(600):
        sched.step()
        tcs.append(sched.get_tc())

    # Monotone non-decreasing
    for i in range(1, len(tcs)):
        assert tcs[i] >= tcs[i - 1] - 1e-14, \
            f'TC decreased at step {i}: {tcs[i]} < {tcs[i - 1]}'

    # Capped at TC_max
    assert max(tcs) <= sched.tc_max + 1e-14, \
        f'TC exceeded tc_max: {max(tcs)}'

    # After passing S=500 steps the plateau is exactly TC_max
    assert np.isclose(tcs[-1], sched.tc_max), \
        f'TC did not reach tc_max at end: {tcs[-1]}'


def test_tc_phase1_convexity():
    # Phase 1: a=1, TC is strictly linear in global step
    episode_length = 100
    sched = TCScheduler(episode_length=episode_length, phase=1)
    assert sched.a == 1.0

    S = sched.S  # 100 * episode_length = 10_000
    # For x <= S: TC(x) = (tc_max / S) * x
    x1, x2 = S // 4, S // 2
    sched._global_step = x1
    tc1 = sched.get_tc()
    sched._global_step = x2
    tc2 = sched.get_tc()

    # Linear: TC(x2) / TC(x1) == x2 / x1
    assert np.isclose(tc2 / tc1, x2 / x1, rtol=1e-9), \
        f'Phase 1 TC not linear: ratio {tc2 / tc1} vs expected {x2 / x1}'


def test_tc_phase23_convexity():
    # Phases 2 and 3 use a=0.45 (concave power curve)
    sched2 = TCScheduler(episode_length=100, phase=2)
    sched3 = TCScheduler(episode_length=100, phase=3)
    assert sched2.a == 0.45, f'Phase 2 a expected 0.45, got {sched2.a}'
    assert sched3.a == 0.45, f'Phase 3 a expected 0.45, got {sched3.a}'

    # a=0.45 means sub-linear (concave): TC grows slower than linearly
    # Verify TC(2x) < 2 * TC(x) for some x <= S
    S = sched2.S
    x = S // 4
    sched2._global_step = x
    tc_x = sched2.get_tc()
    sched2._global_step = 2 * x
    tc_2x = sched2.get_tc()
    # Concave: tc_2x / tc_x == (2x)^0.45 / x^0.45 = 2^0.45 ≈ 1.366 < 2
    expected_ratio = (2 * x) ** 0.45 / x ** 0.45
    assert np.isclose(tc_2x / tc_x, expected_ratio, rtol=1e-9), \
        f'Phase 2 TC ratio {tc_2x / tc_x} != expected {expected_ratio}'
    assert tc_2x / tc_x < 2.0, 'Phase 2/3 TC should be concave (sub-linear)'


def test_phase_windows_nonoverlapping():
    for phase, windows in PhaseTrainer.PHASE_WINDOWS.items():
        train_start = pd.Timestamp(windows['train'][0])
        train_end = pd.Timestamp(windows['train'][1])
        val_start = pd.Timestamp(windows['val'][0])
        val_end = pd.Timestamp(windows['val'][1])
        test_start = pd.Timestamp(windows['test'][0])
        test_end = pd.Timestamp(windows['test'][1])

        assert train_end <= val_start, \
            f'Phase {phase}: train end {train_end} overlaps val start {val_start}'
        assert val_end <= test_start, \
            f'Phase {phase}: val end {val_end} overlaps test start {test_start}'
        assert train_start < train_end, \
            f'Phase {phase}: train window is empty or reversed'
        assert val_start < val_end, \
            f'Phase {phase}: val window is empty or reversed'
        assert test_start < test_end, \
            f'Phase {phase}: test window is empty or reversed'
