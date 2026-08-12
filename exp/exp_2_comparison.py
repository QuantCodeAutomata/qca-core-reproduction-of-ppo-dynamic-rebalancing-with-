"""Experiment 2 — Reward-Function Comparison (4 methods across 3 phases).

Methods
    static_6040       Static 60/40 portfolio (Strategy 2 held continuously)
    ppo_diff_sharpe   PPO with Differential Sharpe reward  + BB + TC
    ppo_emb_drawdown  PPO with Embedded Drawdown reward    + BB + TC
    ppo_regret        PPO with Oracle Regret reward        + BB + TC

Paper annual-return anchors (test periods)
    Phase 1:  DiffSharpe=0.053  EmbDrwdn=0.041  Regret=0.064  BM=0.056
    Phase 2:  DiffSharpe=0.093  EmbDrwdn=0.072  Regret=0.128  BM=0.105
    Phase 3:  DiffSharpe=-0.024 EmbDrwdn=-0.038 Regret=-0.007 BM=-0.026
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

# ── paper anchors ─────────────────────────────────────────────────────────────
PAPER_ANCHORS = {
    1: {'static_6040': 0.056, 'ppo_diff_sharpe': 0.053,
        'ppo_emb_drawdown': 0.041, 'ppo_regret': 0.064},
    2: {'static_6040': 0.105, 'ppo_diff_sharpe': 0.093,
        'ppo_emb_drawdown': 0.072, 'ppo_regret': 0.128},
    3: {'static_6040': -0.026, 'ppo_diff_sharpe': -0.024,
        'ppo_emb_drawdown': -0.038, 'ppo_regret': -0.007},
}

METHOD_CONFIGS = [
    ('ppo_diff_sharpe',  {'reward_type': 'diff_sharpe',      'use_bootstrap': True, 'use_tc_schedule': True}),
    ('ppo_emb_drawdown', {'reward_type': 'embedded_drawdown', 'use_bootstrap': True, 'use_tc_schedule': True}),
    ('ppo_regret',       {'reward_type': 'regret',            'use_bootstrap': True, 'use_tc_schedule': True}),
]

DISPLAY_NAMES = {
    'static_6040':      'Static 60/40',
    'ppo_diff_sharpe':  'PPO-DiffSharpe',
    'ppo_emb_drawdown': 'PPO-EmbDrwdn',
    'ppo_regret':       'PPO-Regret',
}
COLORS = {
    'static_6040':      'darkorange',
    'ppo_diff_sharpe':  'steelblue',
    'ppo_emb_drawdown': 'forestgreen',
    'ppo_regret':       'crimson',
}


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print('Loading data …')
    features_df, strategy_returns_df = load_data()
    print(f'  features shape : {features_df.shape}')

    all_results = {}   # method_name -> {phase -> {train, val, test, model?}}

    # ── static 60/40: no training needed ──────────────────────────────────────
    # Benchmark returns are simply strat_2 returns in each test window,
    # computed via _slice_data after we have any trainer instance.
    # We build a dummy trainer to use _slice_data conveniently.
    _ref_trainer = PhaseTrainer(
        features_df=features_df,
        strategy_returns_df=strategy_returns_df,
        reward_type='return_only',
        use_bootstrap=False,
        use_tc_schedule=False,
        total_timesteps_per_phase=1,
        checkpoint_dir=os.path.join(RESULTS_DIR, 'checkpoints_exp2_ref'),
    )

    bm_phase_metrics = {}
    for phase in [1, 2, 3]:
        _, test_rets = _ref_trainer._slice_data(*PhaseTrainer.PHASE_WINDOWS[phase]['test'])
        bm_rets = test_rets['strat_2'].reset_index(drop=True)
        bm_phase_metrics[phase] = sanitize_metrics(compute_metrics_dict(bm_rets))
    all_results['static_6040'] = {p: {'test': bm_phase_metrics[p]} for p in [1, 2, 3]}

    # ── PPO methods ───────────────────────────────────────────────────────────
    for method_name, cfg in METHOD_CONFIGS:
        print(f'\n=== Training {DISPLAY_NAMES[method_name]} ===')
        trainer = PhaseTrainer(
            features_df=features_df,
            strategy_returns_df=strategy_returns_df,
            total_timesteps_per_phase=8_192,
            seed=42,
            checkpoint_dir=os.path.join(RESULTS_DIR, f'checkpoints_exp2_{method_name}'),
            **cfg,
        )
        raw_results = trainer.run_all_phases()

        method_data = {}
        for phase in [1, 2, 3]:
            method_data[phase] = metrics_from_ppo_results(raw_results[phase])
            # keep model for potential future use; don't serialise
            method_data[phase]['_model'] = raw_results[phase]['model']
            method_data[phase]['_trainer'] = trainer
        all_results[method_name] = method_data

    # ── Build comparison table ────────────────────────────────────────────────
    comparison = {}
    for phase in [1, 2, 3]:
        comparison[phase] = {}
        for method_name in ['static_6040', 'ppo_diff_sharpe', 'ppo_emb_drawdown', 'ppo_regret']:
            test_m = all_results[method_name][phase]['test']
            comparison[phase][method_name] = test_m

    # Print
    for phase in [1, 2, 3]:
        windows = PhaseTrainer.PHASE_WINDOWS[phase]
        print(f'\n=== Phase {phase} test ({windows["test"][0]} → {windows["test"][1]}) ===')
        print(f'{"Method":<20} {"Ann.Ret":>10} {"MDD":>10} {"Sharpe":>10} '
              f'{"Paper Ann.Ret":>14}')
        print('-' * 68)
        for mname in ['static_6040', 'ppo_diff_sharpe', 'ppo_emb_drawdown', 'ppo_regret']:
            m = comparison[phase][mname]
            ann = m.get('annual_return')
            mdd = m.get('max_drawdown')
            sr  = m.get('sharpe')
            paper = PAPER_ANCHORS[phase][mname]
            ann_s = f'{ann:>10.4f}' if ann is not None else f'{"N/A":>10}'
            mdd_s = f'{mdd:>10.4f}' if mdd is not None else f'{"N/A":>10}'
            sr_s  = f'{sr:>10.4f}'  if sr  is not None else f'{"N/A":>10}'
            print(f'{DISPLAY_NAMES[mname]:<20} {ann_s} {mdd_s} {sr_s} {paper:>14.4f}')

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_data = {
        str(phase): {
            mname: comparison[phase][mname]
            for mname in comparison[phase]
        }
        for phase in comparison
    }
    save_json(json_data, os.path.join(RESULTS_DIR, 'exp2_comparison.json'))

    # ── Bar chart: test annual return per phase ────────────────────────────────
    methods_ordered = ['static_6040', 'ppo_diff_sharpe', 'ppo_emb_drawdown', 'ppo_regret']
    n_methods = len(methods_ordered)
    x = np.arange(n_methods)
    width = 0.22

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    for ax, phase in zip(axes, [1, 2, 3]):
        windows = PhaseTrainer.PHASE_WINDOWS[phase]
        vals = []
        for mname in methods_ordered:
            ann = comparison[phase][mname].get('annual_return')
            vals.append(ann if ann is not None else 0.0)

        paper_vals = [PAPER_ANCHORS[phase][m] for m in methods_ordered]

        bars = ax.bar(x, vals, width=0.5,
                      color=[COLORS[m] for m in methods_ordered], alpha=0.75)
        ax.scatter(x, paper_vals, marker='D', color='black', zorder=5,
                   s=40, label='Paper target')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY_NAMES[m] for m in methods_ordered],
                           rotation=20, ha='right', fontsize=8)
        ax.set_title(f'Phase {phase}\n({windows["test"][0]} → {windows["test"][1]})',
                     fontsize=9)
        ax.set_ylabel('Annual Return')
        ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Experiment 2: Test-Period Annual Return by Reward Function', fontsize=11)
    fig.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'exp2_comparison.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f'Saved {plot_path}')

    print('\nExperiment 2 complete.')
