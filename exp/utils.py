"""Shared utilities for all experiment scripts."""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is importable regardless of working directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.environment import PortfolioEnv
from src.features import FeatureBuilder
from src.metrics import (
    compute_all_metrics,
    compute_annual_return,
    compute_calmar_ratio,
    compute_max_drawdown,
    compute_omega_ratio,
    compute_portfolio_returns,
    compute_sharpe_ratio,
)

RAW_PRICES_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw_prices.csv')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load features and strategy returns from raw_prices.csv.

    Returns:
        (features_df, strategy_returns_df) — both have DatetimeIndex.
    """
    if not os.path.exists(RAW_PRICES_PATH):
        print(f'ERROR: {RAW_PRICES_PATH} not found.')
        print('Run:  python data/download.py')
        sys.exit(1)
    builder = FeatureBuilder(RAW_PRICES_PATH)
    features_df = builder.build()
    strategy_returns_df = builder.get_strategy_returns()
    return features_df, strategy_returns_df


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def run_inference_for_plot(model, features: pd.DataFrame,
                           strategy_rets: pd.DataFrame,
                           deterministic: bool = True):
    """Run PPO inference and return weight decisions and net returns.

    Args:
        model: Trained stable-baselines3 PPO model.
        features: Sliced features DataFrame (reset integer index).
        strategy_rets: Sliced strategy returns DataFrame (reset integer index).
        deterministic: Use deterministic (mean) action when True.

    Returns:
        weights_df: DataFrame (T-1, 3) with columns ['s1', 's2', 's3'].
        net_rets:   pd.Series of length T-1 with step-level net returns.
    """
    env = PortfolioEnv(
        features_df=features,
        strategy_returns_df=strategy_rets,
        reward_type='return_only',
        oracle=None,
        tc_schedule=None,
        split_end_idx=len(features) - 1,
    )
    obs, _ = env.reset()
    all_weights, all_net_rets = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        all_weights.append(info['weights'])
        all_net_rets.append(info['net_ret'])

    weights_df = pd.DataFrame(all_weights, columns=['s1', 's2', 's3'])
    net_rets = pd.Series(all_net_rets, name='net_ret')
    return weights_df, net_rets


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_metrics_dict(returns: pd.Series) -> dict:
    """Compute standard performance metrics for a daily returns series."""
    return {
        'annual_return': compute_annual_return(returns),
        'max_drawdown': compute_max_drawdown(returns),
        'sharpe': compute_sharpe_ratio(returns),
        'calmar': compute_calmar_ratio(returns),
        'omega': compute_omega_ratio(returns),
    }


def sanitize_metrics(d: dict) -> dict:
    """Replace non-finite floats with None for JSON serialisation."""
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            out[k] = None
        elif isinstance(v, np.floating):
            fv = float(v)
            out[k] = None if (np.isnan(fv) or np.isinf(fv)) else fv
        else:
            out[k] = v
    return out


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and non-finite floats."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            fv = float(obj)
            return None if (np.isnan(fv) or np.isinf(fv)) else fv
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(data, path: str) -> None:
    """Save *data* to *path* with NumpyEncoder and indent=2."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    print(f'Saved {path}')


# ---------------------------------------------------------------------------
# NAV / plotting helpers
# ---------------------------------------------------------------------------

def compute_nav(returns: pd.Series) -> pd.Series:
    """Compute cumulative NAV starting from 1.0."""
    return (1.0 + returns).cumprod()


def metrics_from_ppo_results(phase_result: dict) -> dict:
    """Extract train/val/test metric dicts from a phase result dict."""
    out = {}
    for split in ('train', 'val', 'test'):
        raw = phase_result[split]
        out[split] = sanitize_metrics({k: v for k, v in raw.items() if k != 'label'})
    return out
