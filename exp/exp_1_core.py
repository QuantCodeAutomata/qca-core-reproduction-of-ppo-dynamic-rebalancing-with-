"""Experiment 1 — Core PPO with Oracle Regret Reward (BB + TC + Regret).

Trains the full-method PPO across all three phases and compares test-period
performance against the static 60/40 benchmark (Strategy 2).

Paper annual-return targets
    Phase 1 test:  60/40 = 0.056,  Regret-PPO = 0.064
    Phase 2 test:  60/40 = 0.105,  Regret-PPO = 0.128
    Phase 3 test:  60/40 = -0.026, Regret-PPO = -0.007
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# ── project root on path ────────────────────────────────────────────────────
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

# ── paper targets ────────────────────────────────────────────────────────────
PAPER_TARGETS = {
    1: {'benchmark': 0.056,  'regret': 0.064},
    2: {'benchmark': 0.105,  'regret': 0.128},
    3: {'benchmark': -0.026, 'regret': -0.007},
}


# ── helpers ──────────────────────────────────────────────────────────────────

def benchmark_returns_for_phase(trainer: PhaseTrainer, phase: int) -> pd.Series:
    """Return benchmark (Strategy 2) net returns for a phase's test window."""
    windows = PhaseTrainer.PHASE_WINDOWS[phase]
    _, test_rets = trainer._slice_data(*windows['test'])
    return test_rets['strat_2'].reset_index(drop=True)


def print_comparison_table(all_metrics: dict) -> None:
    header = f'{"Metric":<22} {"PPO-Regret":>12} {"Benchmark":>12} ' \
             f'{"Paper PPO":>12} {"Paper BM":>12}'
    for phase in [1, 2, 3]:
        windows = PhaseTrainer.PHASE_WINDOWS[phase]
        test_start, test_end = windows['test']
        print(f'\n=== Phase {phase} test ({test_start} → {test_end}) ===')
        print(header)
        print('-' * 72)
        ppo = all_metrics[phase]['test']
        bm  = all_metrics[phase]['benchmark_test']
        for metric in ('annual_return', 'max_drawdown', 'sharpe', 'calmar', 'omega'):
            pv = ppo.get(metric)
            bv = bm.get(metric)
            pp = PAPER_TARGETS[phase]['regret']   if metric == 'annual_return' else float('nan')
            pb = PAPER_TARGETS[phase]['benchmark'] if metric == 'annual_return' else float('nan')
            pv_s = f'{pv:>12.4f}' if pv is not None else f'{"N/A":>12}'
            bv_s = f'{bv:>12.4f}' if bv is not None else f'{"N/A":>12}'
            print(f'{metric:<22} {pv_s} {bv_s} {pp:>12.4f} {pb:>12.4f}')


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load data
    print('Loading data …')
    features_df, strategy_returns_df = load_data()
    print(f'  features shape : {features_df.shape}')
    print(f'  date range     : {features_df.index[0].date()} → {features_df.index[-1].date()}')

    # 2. Train full-method PPO (BB + TC + Regret)
    trainer = PhaseTrainer(
        features_df=features_df,
        strategy_returns_df=strategy_returns_df,
        reward_type='regret',
        use_bootstrap=True,
        use_tc_schedule=True,
        total_timesteps_per_phase=8_192,
        seed=42,
        checkpoint_dir=os.path.join(RESULTS_DIR, 'checkpoints_exp1'),
    )
    results = trainer.run_all_phases()

    # 3. Collect metrics for all phases
    all_metrics = {}
    for phase in [1, 2, 3]:
        bm_rets = benchmark_returns_for_phase(trainer, phase)
        all_metrics[phase] = {
            **metrics_from_ppo_results(results[phase]),
            'benchmark_test': sanitize_metrics(compute_metrics_dict(bm_rets)),
        }

    # 4. Print comparison table
    print_comparison_table(all_metrics)

    # 5. Save JSON
    save_json({str(p): v for p, v in all_metrics.items()},
              os.path.join(RESULTS_DIR, 'exp1_metrics.json'))

    # 6 & 7. Plots per phase
    for phase in [1, 2, 3]:
        windows = PhaseTrainer.PHASE_WINDOWS[phase]
        test_start, test_end = windows['test']
        model = results[phase]['model']

        # Inference on test window
        test_feat, test_rets = trainer._slice_data(*windows['test'])
        weights_df, net_rets = run_inference_for_plot(model, test_feat, test_rets)

        # Benchmark returns (same length alignment)
        bm_rets = benchmark_returns_for_phase(trainer, phase)
        n = min(len(net_rets), len(bm_rets))
        ppo_nav = compute_nav(net_rets.iloc[:n])
        bm_nav  = compute_nav(bm_rets.iloc[:n])
        x_days  = range(n)

        # ── NAV plot ──────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_days, ppo_nav.values, label='PPO-Regret', color='steelblue', linewidth=1.5)
        ax.plot(x_days, bm_nav.values,  label='60/40 Benchmark',
                color='darkorange', linestyle='--', linewidth=1.5)
        ax.set_title(f'Phase {phase} Test-Period Cumulative NAV  ({test_start} → {test_end})')
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Cumulative NAV')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        nav_path = os.path.join(RESULTS_DIR, f'exp1_phase{phase}_nav.png')
        fig.savefig(nav_path, dpi=150)
        plt.close(fig)
        print(f'Saved {nav_path}')

        # ── Weight-trajectory plot ────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.stackplot(
            range(len(weights_df)),
            weights_df['s1'].values,
            weights_df['s2'].values,
            weights_df['s3'].values,
            labels=['Strat 1 — Dev Eq (EFA)',
                    'Strat 2 — 60/40 blend',
                    'Strat 3 — Govies (TLT)'],
            alpha=0.75,
        )
        ax.set_title(f'Phase {phase} Test-Period Weight Allocation  ({test_start} → {test_end})')
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Portfolio Weight')
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        weights_path = os.path.join(RESULTS_DIR, f'exp1_phase{phase}_weights.png')
        fig.savefig(weights_path, dpi=150)
        plt.close(fig)
        print(f'Saved {weights_path}')

    # 8. Summary vs paper targets
    print('\n=== Summary: Annual Return vs Paper Targets ===')
    print(f'{"Phase":<8} {"PPO Annual Ret":>16} {"BM Annual Ret":>16} '
          f'{"Paper PPO":>12} {"Paper BM":>12}')
    print('-' * 68)
    for phase in [1, 2, 3]:
        ppo_ann = all_metrics[phase]['test'].get('annual_return')
        bm_ann  = all_metrics[phase]['benchmark_test'].get('annual_return')
        pv_s = f'{ppo_ann:>16.4f}' if ppo_ann is not None else f'{"N/A":>16}'
        bv_s = f'{bm_ann:>16.4f}'  if bm_ann  is not None else f'{"N/A":>16}'
        print(f'{phase:<8} {pv_s} {bv_s} '
              f'{PAPER_TARGETS[phase]["regret"]:>12.4f} '
              f'{PAPER_TARGETS[phase]["benchmark"]:>12.4f}')

    print('\nExperiment 1 complete.')
