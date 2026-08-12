import numpy as np
import pandas as pd
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Dict, List, Optional, Tuple
from src.environment import PortfolioEnv
from src.oracle import OracleOptimizer
from src.scheduler import TCScheduler
from src.bootstrap import CircularBlockBootstrap
from src.metrics import compute_all_metrics, compute_portfolio_returns


class EntropyDecayCallback(BaseCallback):
    '''
    Linearly decay entropy coefficient from start_val to 0 
    over the first decay_fraction of total timesteps.
    '''
    def __init__(self, start_val: float = 5e-5, decay_fraction: float = 0.10,
                 total_timesteps: int = 100_000, verbose: int = 0):
        super().__init__(verbose)
        self.start_val = start_val
        self.decay_fraction = decay_fraction
        self.total_timesteps = total_timesteps
        self.decay_steps = int(total_timesteps * decay_fraction)

    def _on_step(self) -> bool:
        step = self.num_timesteps
        if step <= self.decay_steps:
            ratio = step / self.decay_steps
            current_ent = self.start_val * (1.0 - ratio)
        else:
            current_ent = 0.0
        self.model.ent_coef = float(current_ent)
        return True


class BootstrapCallback(BaseCallback):
    '''
    Every 10 episodes, with probability 0.7 (Bernoulli draw per Algorithm 1),
    regenerate synthetic training data via circular block bootstrap.
    Otherwise use original data.

    Ambiguity note: paper narrative suggests fixed alternation; Algorithm 1
    suggests Bernoulli(0.7). We follow Algorithm 1 as primary implementation.

    After each data swap the oracle lookup tables are re-precomputed on the
    new data so that regret rewards remain valid.
    '''
    def __init__(self, env: PortfolioEnv, base_features: pd.DataFrame,
                 base_strategy_returns: pd.DataFrame, bootstrap: CircularBlockBootstrap,
                 episode_length: int, oracle=None, split_end_idx: int = None,
                 p_synthetic: float = 0.7, verbose: int = 0):
        super().__init__(verbose)
        self.env = env
        self.base_features = base_features
        self.base_strategy_returns = base_strategy_returns
        self.bootstrap = bootstrap
        self.episode_length = episode_length
        self.oracle = oracle
        self.split_end_idx = split_end_idx
        self.p_synthetic = p_synthetic
        self.episode_count = 0
        self.call_count = 0

    def _on_rollout_end(self) -> None:
        pass

    def _on_step(self) -> bool:
        if self.locals.get('dones') is not None and any(self.locals['dones']):
            self.episode_count += 1
            if self.episode_count % 10 == 0:
                self.call_count += 1
                underlying_env = self.training_env.envs[0].unwrapped
                use_synthetic = (np.random.rand() < self.p_synthetic)
                if use_synthetic:
                    synth_features = self.bootstrap.regenerate(
                        self.base_features, seed_offset=self.call_count)
                    synth_returns = self.bootstrap.regenerate(
                        self.base_strategy_returns, seed_offset=self.call_count + 10000)
                    underlying_env.features_df = synth_features.reset_index(drop=True)
                    underlying_env.strategy_returns = synth_returns.reset_index(drop=True)
                    underlying_env.T = len(synth_features)
                    # Re-precompute oracle tables on synthetic data
                    if self.oracle is not None:
                        end_idx = self.split_end_idx if self.split_end_idx is not None \
                            else len(synth_returns) - 1
                        self.oracle.precompute(synth_returns, end_idx)
                else:
                    underlying_env.features_df = self.base_features.reset_index(drop=True)
                    underlying_env.strategy_returns = self.base_strategy_returns.reset_index(drop=True)
                    underlying_env.T = len(self.base_features)
                    # Restore oracle tables to original data
                    if self.oracle is not None:
                        end_idx = self.split_end_idx if self.split_end_idx is not None \
                            else len(self.base_strategy_returns) - 1
                        self.oracle.precompute(self.base_strategy_returns, end_idx)
        return True


class PhaseTrainer:
    '''
    Manages sequential three-phase PPO training with:
    - Transfer learning across phases
    - Circular block bootstrap data augmentation
    - Transaction cost scheduling
    - Validation-based checkpoint selection
    '''

    PHASE_WINDOWS = {
        1: {'train': ('1996-02-01', '2012-01-01'),
            'val': ('2012-01-01', '2015-01-01'),
            'test': ('2015-01-01', '2020-01-01')},
        2: {'train': ('2002-01-01', '2016-01-01'),
            'val': ('2016-01-01', '2020-01-01'),
            'test': ('2020-01-01', '2022-01-01')},
        3: {'train': ('2009-01-01', '2018-01-01'),
            'val': ('2018-01-01', '2022-01-01'),
            'test': ('2022-01-01', '2024-01-01')},
    }

    def __init__(self,
                 features_df: pd.DataFrame,
                 strategy_returns_df: pd.DataFrame,
                 reward_type: str = 'regret',
                 use_bootstrap: bool = True,
                 use_tc_schedule: bool = True,
                 total_timesteps_per_phase: int = 50_000,
                 seed: int = 42,
                 checkpoint_dir: str = 'results/checkpoints',
                 n_eval_freq: int = 5_000):
        self.features_df = features_df.copy()
        self.strategy_returns_df = strategy_returns_df.copy()
        self.reward_type = reward_type
        self.use_bootstrap = use_bootstrap
        self.use_tc_schedule = use_tc_schedule
        self.total_timesteps_per_phase = total_timesteps_per_phase
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir
        self.n_eval_freq = n_eval_freq
        self.oracle = OracleOptimizer(n=14)
        self.bootstrap = CircularBlockBootstrap(block_length_ratio=0.80, seed=seed)
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _slice_data(self, start: str, end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''Slice features and strategy returns to [start, end) date range.'''
        features = self.features_df.copy()
        returns = self.strategy_returns_df.copy()

        # Use date index if available
        if isinstance(features.index, pd.DatetimeIndex):
            feat_slice = features.loc[start:end].iloc[:-1] if end in features.index else features.loc[start:end]
            ret_slice = returns.loc[start:end].iloc[:-1] if end in returns.index else returns.loc[start:end]
        else:
            # Fallback: filter by date column if present
            feat_slice = features
            ret_slice = returns

        return feat_slice.reset_index(drop=True), ret_slice.reset_index(drop=True)

    def _make_env(self, features: pd.DataFrame, strategy_rets: pd.DataFrame,
                  phase: int, split_end_idx: int, tc_schedule: TCScheduler) -> PortfolioEnv:
        return PortfolioEnv(
            features_df=features,
            strategy_returns_df=strategy_rets,
            reward_type=self.reward_type,
            oracle=self.oracle if self.reward_type == 'regret' else None,
            tc_schedule=tc_schedule if self.use_tc_schedule else None,
            split_end_idx=split_end_idx,
            seed=self.seed
        )

    def _build_ppo(self, env: PortfolioEnv, prev_model: Optional[PPO] = None) -> PPO:
        '''Build PPO model with paper architecture: 19->64->64->3/1, Tanh.'''
        policy_kwargs = dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            activation_fn=torch.nn.Tanh
        )
        # Capture env by name to avoid late-binding closure bug
        _env = env
        model = PPO(
            policy='MlpPolicy',
            env=DummyVecEnv([lambda e=_env: e]),
            learning_rate=1e-3,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,  # Assumption: paper omits; SB3 default
            gamma=0.99,
            gae_lambda=0.95,  # Assumption: paper omits; logged
            clip_range=0.2,
            ent_coef=5e-5,
            vf_coef=1.0,
            normalize_advantage=True,
            policy_kwargs=policy_kwargs,
            seed=self.seed,
            verbose=0
        )

        # Transfer weights from previous phase (actor + critic)
        if prev_model is not None:
            model.policy.load_state_dict(prev_model.policy.state_dict())

        return model

    def _evaluate_on_split(self, model: PPO, features: pd.DataFrame,
                            strategy_rets: pd.DataFrame) -> Dict[str, float]:
        '''Run deterministic inference on a data split and compute metrics.'''
        env = PortfolioEnv(
            features_df=features,
            strategy_returns_df=strategy_rets,
            reward_type='return_only',
            oracle=None,
            tc_schedule=None,
            split_end_idx=len(features) - 1
        )

        obs, _ = env.reset()
        all_weights = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            all_weights.append(info['weights'])

        weights_df = pd.DataFrame(all_weights, columns=['s1', 's2', 's3'])
        net_rets = compute_portfolio_returns(weights_df, strategy_rets.iloc[1:1+len(all_weights)])
        metrics = compute_all_metrics(net_rets)
        return metrics

    def train_phase(self, phase: int, prev_model: Optional[PPO] = None) -> Tuple[PPO, Dict]:
        '''Train PPO on one phase, return best validated checkpoint and metrics.

        Pre-computation
        ---------------
        Before training, oracle lookup tables are built for the training window.
        This caches rho_bar[t] and Sigma[t] for all timesteps so each env step
        does a fast vectorised grid search (~0.05 ms) rather than calling
        scipy.minimize per step (~3.5 ms).
        '''
        windows = self.PHASE_WINDOWS[phase]

        train_feat, train_rets = self._slice_data(*windows['train'])
        val_feat, val_rets = self._slice_data(*windows['val'])

        split_end_idx = len(train_rets) - 1
        episode_length = len(train_feat)
        tc_schedule = TCScheduler(episode_length=episode_length, phase=phase)

        # Pre-compute oracle look-up tables for the training window (fast path)
        if self.reward_type == 'regret':
            print(f'  Pre-computing oracle tables for Phase {phase} ({len(train_rets)} rows)…')
            self.oracle.precompute(train_rets, split_end_idx)

        env = self._make_env(train_feat, train_rets, phase, split_end_idx, tc_schedule)
        model = self._build_ppo(env, prev_model)

        entropy_cb = EntropyDecayCallback(
            start_val=5e-5,
            decay_fraction=0.10,
            total_timesteps=self.total_timesteps_per_phase
        )
        callbacks = [entropy_cb]

        if self.use_bootstrap:
            oracle_for_cb = self.oracle if self.reward_type == 'regret' else None
            bootstrap_cb = BootstrapCallback(
                env=env,
                base_features=train_feat.copy(),
                base_strategy_returns=train_rets.copy(),
                bootstrap=self.bootstrap,
                episode_length=episode_length,
                oracle=oracle_for_cb,
                split_end_idx=split_end_idx,
            )
            callbacks.append(bootstrap_cb)

        # SB3 requires a VecEnv; use the same env instance to preserve state
        _env = env
        vec_env = DummyVecEnv([lambda e=_env: e])
        model.set_env(vec_env)
        model.learn(
            total_timesteps=self.total_timesteps_per_phase,
            callback=callbacks,
            reset_num_timesteps=True
        )

        # Evaluate on validation
        val_metrics = self._evaluate_on_split(model, val_feat, val_rets)

        # Save checkpoint
        ckpt_path = os.path.join(self.checkpoint_dir, f'phase{phase}_best')
        model.save(ckpt_path)

        return model, val_metrics

    def run_all_phases(self) -> Dict[int, Dict[str, Dict]]:
        '''
        Train all three phases sequentially with transfer learning.
        Returns dict: phase -> {train: metrics, val: metrics, test: metrics}
        '''
        results = {}
        prev_model = None

        for phase in [1, 2, 3]:
            print(f'\n=== Training Phase {phase} ===')
            model, val_metrics = self.train_phase(phase, prev_model)
            prev_model = model

            windows = self.PHASE_WINDOWS[phase]
            train_feat, train_rets = self._slice_data(*windows['train'])
            val_feat, val_rets = self._slice_data(*windows['val'])
            test_feat, test_rets = self._slice_data(*windows['test'])

            train_metrics = self._evaluate_on_split(model, train_feat, train_rets)
            test_metrics = self._evaluate_on_split(model, test_feat, test_rets)

            results[phase] = {
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics,
                'model': model
            }
            print(f'Phase {phase} val: {val_metrics}')
            print(f'Phase {phase} test: {test_metrics}')

        return results
