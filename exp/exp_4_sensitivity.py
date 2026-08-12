"""Experiment 4 — Sensitivity Analysis on Unspecified Paper Choices.

Sensitivity axes (one parameter varied at a time; others held at defaults):
    bootstrap_schedule  'algorithm1' (Bernoulli 0.7) vs 'alternation' (strict)
    block_frac          [0.70, 0.80, 0.90]
    gae_lambda          [0.95, 0.97, 0.99]
    n_epochs            [5, 10, 20]
    deterministic       [True, False]

Defaults used when an axis is NOT being varied:
    bootstrap_schedule='algorithm1', block_frac=0.80,
    gae_lambda=0.97, n_epochs=10, deterministic=True

All runs use: reward_type='regret', use_bootstrap=True, use_tc_schedule=True.
"""

import os
import sys
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from exp.utils import (
    RESULTS_DIR,
    compute_metrics_dict,
    compute_nav,
    load_data,
    metrics_from_ppo_results,
    run_inference_for_plot,
    save_json,
    sanitize_metrics,
)
from src.bootstrap import CircularBlockBootstrap
from src.environment import PortfolioEnv
from src.metrics import compute_all_metrics, compute_portfolio_returns
from src.scheduler import TCScheduler
from src.trainer import BootstrapCallback, EntropyDecayCallback, PhaseTrainer

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    bootstrap_schedule='algorithm1',
    block_frac=0.80,
    gae_lambda=0.97,
    n_epochs=10,
    deterministic=True,
)

# ── Sensitivity axes ──────────────────────────────────────────────────────────
AXES = {
    'bootstrap_schedule': ['algorithm1', 'alternation'],
    'block_frac':         [0.70, 0.80, 0.90],
    'gae_lambda':         [0.95, 0.97, 0.99],
    'n_epochs':           [5, 10, 20],
    'deterministic':      [True, False],
}


# ── Strict-alternation bootstrap callback ─────────────────────────────────────

class AlternatingBootstrapCallback(BootstrapCallback):
    """Every 10 episodes, strictly alternate between synthetic and real data.

    After each swap, oracle tables are re-precomputed (inherits oracle +
    split_end_idx from BootstrapCallback.__init__).
    """

    def _on_step(self) -> bool:
        if self.locals.get('dones') is not None and any(self.locals['dones']):
            self.episode_count += 1
            if self.episode_count % 10 == 0:
                self.call_count += 1
                use_synthetic = (self.call_count % 2 == 1)   # 1→synth, 2→real, 3→synth …
                underlying_env = self.training_env.envs[0].unwrapped
                if use_synthetic:
                    synth_feat = self.bootstrap.regenerate(
                        self.base_features, seed_offset=self.call_count)
                    synth_rets = self.bootstrap.regenerate(
                        self.base_strategy_returns, seed_offset=self.call_count + 10_000)
                    underlying_env.features_df = synth_feat.reset_index(drop=True)
                    underlying_env.strategy_returns = synth_rets.reset_index(drop=True)
                    underlying_env.T = len(synth_feat)
                    if self.oracle is not None:
                        end_idx = self.split_end_idx if self.split_end_idx is not None \
                            else len(synth_rets) - 1
                        self.oracle.precompute(synth_rets, end_idx)
                else:
                    underlying_env.features_df = self.base_features.reset_index(drop=True)
                    underlying_env.strategy_returns = self.base_strategy_returns.reset_index(drop=True)
                    underlying_env.T = len(self.base_features)
                    if self.oracle is not None:
                        end_idx = self.split_end_idx if self.split_end_idx is not None \
                            else len(self.base_strategy_returns) - 1
                        self.oracle.precompute(self.base_strategy_returns, end_idx)
        return True


# ── Sensitivity-aware PhaseTrainer subclass ───────────────────────────────────

class SensitivityTrainer(PhaseTrainer):
    """PhaseTrainer extended with sensitivity-analysis knobs.

    Extra constructor args (beyond PhaseTrainer):
        block_frac          block_length_ratio for CircularBlockBootstrap
        gae_lambda          GAE lambda for PPO
        n_epochs            PPO gradient epochs per update
        deterministic       deterministic action during evaluation
        bootstrap_schedule  'algorithm1' | 'alternation'
    """

    def __init__(self,
                 features_df: pd.DataFrame,
                 strategy_returns_df: pd.DataFrame,
                 block_frac: float = 0.80,
                 gae_lambda: float = 0.97,
                 n_epochs: int = 10,
                 deterministic: bool = True,
                 bootstrap_schedule: str = 'algorithm1',
                 **kwargs):
        super().__init__(features_df, strategy_returns_df, **kwargs)
        # Override bootstrap with requested block fraction
        self.bootstrap = CircularBlockBootstrap(
            block_length_ratio=block_frac, seed=self.seed)
        self._gae_lambda = gae_lambda
        self._n_epochs = n_epochs
        self._deterministic = deterministic
        self._bootstrap_schedule = bootstrap_schedule

    # -- override PPO build to inject gae_lambda / n_epochs ------------------
    def _build_ppo(self, env: PortfolioEnv,
                   prev_model: Optional[PPO] = None) -> PPO:
        policy_kwargs = dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            activation_fn=torch.nn.Tanh,
        )
        # Capture env by name to avoid late-binding closure bug
        _env = env
        model = PPO(
            policy='MlpPolicy',
            env=DummyVecEnv([lambda e=_env: e]),
            learning_rate=1e-3,
            n_steps=2048,
            batch_size=64,
            n_epochs=self._n_epochs,
            gamma=0.99,
            gae_lambda=self._gae_lambda,
            clip_range=0.2,
            ent_coef=5e-5,
            vf_coef=1.0,
            normalize_advantage=True,
            policy_kwargs=policy_kwargs,
            seed=self.seed,
            verbose=0,
        )
        if prev_model is not None:
            model.policy.load_state_dict(prev_model.policy.state_dict())
        return model

    # -- override evaluation to respect deterministic flag --------------------
    def _evaluate_on_split(self, model: PPO,
                            features: pd.DataFrame,
                            strategy_rets: pd.DataFrame) -> dict:
        env = PortfolioEnv(
            features_df=features,
            strategy_returns_df=strategy_rets,
            reward_type='return_only',
            oracle=None,
            tc_schedule=None,
            split_end_idx=len(features) - 1,
        )
        obs, _ = env.reset()
        all_weights = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=self._deterministic)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            all_weights.append(info['weights'])

        weights_df = pd.DataFrame(all_weights, columns=['s1', 's2', 's3'])
        net_rets = compute_portfolio_returns(
            weights_df, strategy_rets.iloc[1: 1 + len(all_weights)])
        return compute_all_metrics(net_rets)

    # -- override train_phase to add oracle precompute + bootstrap callback ---
    def train_phase(self, phase: int, prev_model=None):
        windows = self.PHASE_WINDOWS[phase]
        train_feat, train_rets = self._slice_data(*windows['train'])
        val_feat, val_rets = self._slice_data(*windows['val'])

        split_end_idx = len(train_rets) - 1
        episode_length = len(train_feat)
        tc_schedule = TCScheduler(episode_length=episode_length, phase=phase)

        # Pre-compute oracle tables before training (fast path)
        if self.reward_type == 'regret':
            print(f'  [Sensitivity] Pre-computing oracle tables for Phase {phase}…')
            self.oracle.precompute(train_rets, split_end_idx)

        env = self._make_env(train_feat, train_rets, phase, split_end_idx, tc_schedule)
        model = self._build_ppo(env, prev_model)

        entropy_cb = EntropyDecayCallback(
            start_val=5e-5,
            decay_fraction=0.10,
            total_timesteps=self.total_timesteps_per_phase,
        )
        callbacks = [entropy_cb]

        if self.use_bootstrap:
            oracle_for_cb = self.oracle if self.reward_type == 'regret' else None
            cb_cls = (AlternatingBootstrapCallback
                      if self._bootstrap_schedule == 'alternation'
                      else BootstrapCallback)
            bootstrap_cb = cb_cls(
                env=env,
                base_features=train_feat.copy(),
                base_strategy_returns=train_rets.copy(),
                bootstrap=self.bootstrap,
                episode_length=episode_length,
                oracle=oracle_for_cb,
                split_end_idx=split_end_idx,
            )
            callbacks.append(bootstrap_cb)

        # Avoid late-binding closure bug
        _env = env
        vec_env = DummyVecEnv([lambda e=_env: e])
        model.set_env(vec_env)
        model.learn(
            total_timesteps=self.total_timesteps_per_phase,
            callback=callbacks,
            reset_num_timesteps=True,
        )

        val_metrics = self._evaluate_on_split(model, val_feat, val_rets)
        ckpt_path = os.path.join(self.checkpoint_dir, f'phase{phase}_best')
        model.save(ckpt_path)
        return model, val_metrics


# ── Helper: run one sensitivity configuration ─────────────────────────────────

def run_sensitivity_config(features_df, strategy_returns_df,
                            config: dict, ckpt_suffix: str) -> dict:
    """Train SensitivityTrainer and return per-phase test metrics."""
    trainer = SensitivityTrainer(
        features_df=features_df,
        strategy_returns_df=strategy_returns_df,
        reward_type='regret',
        use_bootstrap=True,
        use_tc_schedule=True,
        total_timesteps_per_phase=8_192,
        seed=42,
        checkpoint_dir=os.path.join(RESULTS_DIR, f'checkpoints_exp4_{ckpt_suffix}'),
        block_frac=config['block_frac'],
        gae_lambda=config['gae_lambda'],
        n_epochs=config['n_epochs'],
        deterministic=config['deterministic'],
        bootstrap_schedule=config['bootstrap_schedule'],
    )
    raw = trainer.run_all_phases()
    phase_metrics = {}
    for phase in [1, 2, 3]:
        phase_metrics[phase] = sanitize_metrics(
            {k: v for k, v in raw[phase]['test'].items() if k != 'label'})
    return phase_metrics


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print('Loading data …')
    features_df, strategy_returns_df = load_data()
    print(f'  features shape : {features_df.shape}')

    sensitivity_results = {}   # axis -> param_value (str) -> phase -> metrics

    for axis_name, param_values in AXES.items():
        print(f'\n=== Sensitivity axis: {axis_name} ===')
        sensitivity_results[axis_name] = {}

        for val in param_values:
            label = str(val)
            print(f'  {axis_name} = {val}')
            config = {**DEFAULTS, axis_name: val}
            ckpt_suffix = f'{axis_name}_{label}'.replace('.', 'p')
            phase_metrics = run_sensitivity_config(
                features_df, strategy_returns_df, config, ckpt_suffix)
            sensitivity_results[axis_name][label] = {
                str(p): phase_metrics[p] for p in [1, 2, 3]
            }

    # ── Save JSON ──────────────────────────────────────────────────────────────
    save_json(sensitivity_results,
              os.path.join(RESULTS_DIR, 'exp4_sensitivity.json'))

    # ── Plots: one subplot per axis, bars by param value, grouped by phase ────
    n_axes = len(AXES)
    fig, axes_plots = plt.subplots(1, n_axes, figsize=(4 * n_axes, 5))

    phase_colors = {1: '#4878D0', 2: '#EE854A', 3: '#6ACC65'}

    for ax, (axis_name, param_values) in zip(axes_plots, AXES.items()):
        labels = [str(v) for v in param_values]
        n_vals = len(labels)
        n_phases = 3
        group_w = 0.7
        bar_w = group_w / n_phases
        x = np.arange(n_vals)

        for i, phase in enumerate([1, 2, 3]):
            ann_rets = []
            for label in labels:
                m = sensitivity_results[axis_name][label][str(phase)]
                ann_rets.append(m.get('annual_return') or 0.0)
            offset = (i - 1) * bar_w
            ax.bar(x + offset, ann_rets, width=bar_w,
                   color=phase_colors[phase], alpha=0.80,
                   label=f'Phase {phase}')

        ax.axhline(0, color='black', linewidth=0.7, linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel(axis_name, fontsize=8)
        ax.set_ylabel('Test Annual Return')
        ax.set_title(axis_name, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Experiment 4: Sensitivity of Test Annual Return', fontsize=11)
    fig.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'exp4_sensitivity.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f'Saved {plot_path}')

    # ── Print summary table ───────────────────────────────────────────────────
    print('\n=== Sensitivity Summary (test annual return) ===')
    for axis_name, param_values in AXES.items():
        print(f'\n  Axis: {axis_name}')
        print(f'  {"Value":<15} {"Phase 1":>10} {"Phase 2":>10} {"Phase 3":>10}')
        print('  ' + '-' * 48)
        for val in param_values:
            label = str(val)
            row = sensitivity_results[axis_name][label]
            p1 = row['1'].get('annual_return')
            p2 = row['2'].get('annual_return')
            p3 = row['3'].get('annual_return')
            fmt = lambda v: f'{v:>10.4f}' if v is not None else f'{"N/A":>10}'
            print(f'  {label:<15} {fmt(p1)} {fmt(p2)} {fmt(p3)}')

    print('\nExperiment 4 complete.')
