"""Experiment 3 — Ablation of Bootstrap, TC Scheduler, and Reward Function.

Five ablation arms (all use regret reward unless noted):
    1. Regret only          reward='regret',       BB=False, TC=False
    2. BB + Regret          reward='regret',       BB=True,  TC=False
    3. TC + Regret          reward='regret',       BB=False, TC=True
    4. BB + TC + Regret     reward='regret',       BB=True,  TC=True   ← full
    5. BB + TC + Return     reward='return_only',  BB=True,  TC=True
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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
from src.trainer import PhaseTrainer

# ── Ablation arm definitions ──────────────────────────────────────────────────
ARMS = [
    ('regret_only',    {'reward_type': 'regret',      'use_bootstrap': False, 'use_tc_schedule': False}),
    ('bb_regret',      {'reward_type': 'regret',      'use_bootstrap': True,  'use_tc_schedule': False}),
    ('tc_regret',      {'reward_type': 'regret',      'use_bootstrap': False, 'use_tc_schedule': True}),
    ('bb_tc_regret',   {'reward_type': 'regret',      'use_bootstrap': True,  'use_tc_schedule': True}),
    ('bb_tc_return',   {'reward_type': 'return_only', 'use_bootstrap': True,  'use_tc_schedule': True}),
]

DISPLAY_NAMES = {
    'regret_only':  'Regret only',
    'bb_regret':    'BB + Regret',
    'tc_regret':    'TC + Regret',
    'bb_tc_regret': 'BB+TC+Regret (full)',
    'bb_tc_return': 'BB+TC+Return',
}

ARM_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print('Loading data …')
    features_df, strategy_returns_df = load_data()
    print(f'  features shape : {features_df.shape}')

    arm_results   = {}   # arm_name -> {phase -> metrics dicts}
    arm_nav_series = {}  # arm_name -> {phase -> pd.Series}

    for arm_name, cfg in ARMS:
        print(f'\n=== Arm: {DISPLAY_NAMES[arm_name]} ===')
        trainer = PhaseTrainer(
            features_df=features_df,
            strategy_returns_df=strategy_returns_df,
            total_timesteps_per_phase=8_192,
            seed=42,
            checkpoint_dir=os.path.join(RESULTS_DIR, f'checkpoints_exp3_{arm_name}'),
            **cfg,
        )
        raw_results = trainer.run_all_phases()

        arm_results[arm_name] = {}
        arm_nav_series[arm_name] = {}

        for phase in [1, 2, 3]:
            arm_results[arm_name][phase] = metrics_from_ppo_results(raw_results[phase])

            # Collect NAV series for test period
            model = raw_results[phase]['model']
            test_feat, test_rets = trainer._slice_data(*PhaseTrainer.PHASE_WINDOWS[phase]['test'])
            _, net_rets = run_inference_for_plot(model, test_feat, test_rets)
            arm_nav_series[arm_name][phase] = compute_nav(net_rets)

    # ── Print comparison table ─────────────────────────────────────────────────
    for phase in [1, 2, 3]:
        windows = PhaseTrainer.PHASE_WINDOWS[phase]
        print(f'\n=== Phase {phase} test ({windows["test"][0]} → {windows["test"][1]}) ===')
        print(f'{"Arm":<22} {"Ann.Ret (train)":>16} {"Ann.Ret (val)":>14} {"Ann.Ret (test)":>14} '
              f'{"MDD (test)":>12} {"Sharpe (test)":>14}')
        print('-' * 96)
        for arm_name, _ in ARMS:
            m = arm_results[arm_name][phase]
            tr = m['train'].get('annual_return')
            va = m['val'].get('annual_return')
            te = m['test'].get('annual_return')
            md = m['test'].get('max_drawdown')
            sr = m['test'].get('sharpe')

            def fmt(v, w=14):
                return f'{v:{w}.4f}' if v is not None else f'{"N/A":{w}}'

            print(f'{DISPLAY_NAMES[arm_name]:<22} {fmt(tr,16)} {fmt(va,14)} '
                  f'{fmt(te,14)} {fmt(md,12)} {fmt(sr,14)}')

    # ── Save JSON ──────────────────────────────────────────────────────────────
    json_data = {
        arm_name: {
            str(phase): arm_results[arm_name][phase]
            for phase in [1, 2, 3]
        }
        for arm_name, _ in ARMS
    }
    save_json(json_data, os.path.join(RESULTS_DIR, 'exp3_ablation.json'))

    # ── Plot: mean cumulative test NAV, one subplot per phase ──────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, phase in zip(axes, [1, 2, 3]):
        windows = PhaseTrainer.PHASE_WINDOWS[phase]
        for (arm_name, _), color in zip(ARMS, ARM_COLORS):
            nav = arm_nav_series[arm_name][phase]
            ax.plot(nav.values, label=DISPLAY_NAMES[arm_name],
                    color=color, linewidth=1.3)
        ax.axhline(1.0, color='black', linewidth=0.8, linestyle=':', alpha=0.6)
        ax.set_title(f'Phase {phase}\n({windows["test"][0]} → {windows["test"][1]})',
                     fontsize=9)
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Cumulative NAV')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Experiment 3: Ablation — Test-Period Cumulative NAV', fontsize=11)
    fig.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'exp3_ablation.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f'Saved {plot_path}')

    print('\nExperiment 3 complete.')
