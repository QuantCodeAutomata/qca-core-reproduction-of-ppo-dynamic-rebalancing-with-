# Repository Knowledge — PPO Dynamic Portfolio Rebalancing

## Project Overview
This repository implements and reproduces experiments from a paper on PPO-based dynamic portfolio rebalancing using an Oracle-based negative Sharpe regret reward. The system rebalances across three tradable strategy sleeves to improve a static 60/40 benchmark.

## Architecture
- `data/` — Data download (yfinance proxies for Bloomberg indices) and preprocessing
- `src/` — Core implementations: environment, oracle, bootstrap, metrics, trainer
- `exp/` — Four experiment scripts (exp_1 through exp_4)
- `tests/` — pytest tests covering all src modules
- `results/` — Metrics tables, plots, and RESULTS.md

## Key Implementation Choices (Documented Assumptions)

### Data Proxies (Bloomberg → yfinance)
| Paper Ticker | Description | Proxy Used |
|---|---|---|
| MXWOHEUR | MSCI World ex-Aus Hedged EUR (Dev Equities) | EFA (MSCI EAFE ETF, 2001+), SPY gap-fill pre-2001 |
| NDUEEGF | MSCI Emerging Markets | EEM (2003+), VWO pre-fill |
| GOBC | Bloomberg Global Corporate Bond | LQD (iShares Invest. Grade Corp) |
| W3G1 | FTSE World Government Bond | TLT (20+ Year Treasury, 2002+), IEF gap-fill |
| HY Spread | High-Yield Bond Spread | HYG-LQD spread (yfinance) |
| VIX | CBOE VIX | ^VIX |
| MOVE | MOVE Index (bond volatility) | Approximated from TLT volatility |

### PPO Implementation Assumptions
- **GAE lambda**: 0.95 (SB3 default, paper omits this; logged as assumption)
- **PPO epochs per update**: 10 (SB3 default; paper omits; logged as assumption)
- **Action distribution**: Continuous Gaussian in logit space; softmax applied inside env step
- **Inference mode**: Deterministic (mean action) for evaluation
- **Optimizer**: Adam with default betas (paper omits)

### Oracle Optimizer
- Uses `scipy.optimize.minimize` with SLSQP solver
- Fallback: dense simplex grid search (K=3, 100 points)
- Near-singular covariance regularized with epsilon=1e-6

### Circular Block Bootstrap
- Block length: 80% of training period length (mid-range of 70-90%)
- Joint multivariate bootstrap preserving cross-sectional dependence
- Schedule: Algorithm 1 version (Bernoulli(0.7) every 10 episodes)

### Transaction Cost Scheduler
- TC_max = 0.0025, convexity a=1 (Phase 1), a=0.45 (Phases 2&3)
- S = 100 * episode_length
- TC_eval = 0.0025 fixed

### Phase Windows
- Phase 1: train 1996-02-01→2012-01-01, val 2012→2015, test 2015→2020
- Phase 2: train 2002-01-01→2016-01-01, val 2016→2020, test 2020→2022
- Phase 3: train 2009-01-01→2018-01-01, val 2018→2022, test 2022→2024

## Key Dependencies
- stable-baselines3 ≥ 2.9.0 (PPO)
- gymnasium ≥ 1.3.0 (custom environment)
- torch ≥ 2.0 (neural networks)
- cvxpy 1.7.5 (Oracle optimization alternative)
- scipy 1.14.1 (SLSQP oracle, statistics)
- quantstats (performance metrics)
- yfinance (data download)
- pandas 2.2.2, numpy 1.26.4

## Running Experiments
```bash
# Install dependencies
pip install -r requirements.txt

# Run data download (required before any experiment)
python data/download.py

# Run individual experiments
python exp/exp_1_core.py           # Core PPO (BB+TC+Regret) vs 60/40 benchmark
python exp/exp_2_comparison.py     # 4 reward functions compared
python exp/exp_3_ablation.py       # 5 ablation arms
python exp/exp_4_sensitivity.py    # Sensitivity over 5 hyperparameter axes

# Run all experiments in sequence
python exp/run_all.py

# Run tests
pytest tests/ -v
```

## Test Structure
- `tests/test_features.py` — Feature engineering validation
- `tests/test_oracle.py` — Oracle optimizer validation
- `tests/test_bootstrap.py` — Bootstrap data augmentation
- `tests/test_metrics.py` — Performance metrics
- `tests/test_environment.py` — Gymnasium environment
- `tests/test_trainer.py` — Training pipeline components

## Column Naming Convention
- `src/features.py` (FeatureBuilder) produces features_df columns: `strat_1`, `strat_2`, `strat_3`, … (OBS_FEATURE_COLS)
- `src/features.py` also produces strategy_returns_df columns: `strat_1`, `strat_2`, `strat_3`
- `src/environment.py` (PortfolioEnv) reads `strat_1`, `strat_2`, `strat_3` directly from features_df
- Test helpers in all test files use `strat_1`, `strat_2`, `strat_3` for both feature and return columns

## Implemented Modules
- `src/environment.py`: PortfolioEnv — Gymnasium env, 19-dim obs, 3-dim logit action, 4 reward modes
- `src/scheduler.py`: TCScheduler — Power-law TC ramp, a=1 (Phase 1), a=0.45 (Phases 2&3)
- `src/trainer.py`: PhaseTrainer + EntropyDecayCallback + BootstrapCallback — 3-phase PPO pipeline
- `tests/test_environment.py`: 9 pytest tests covering obs shape, softmax, termination, all reward modes, state reset
- `tests/test_trainer.py`: 6 pytest tests covering PPO architecture, entropy decay, TC monotonicity, phase windows
- `exp/utils.py`: Shared helpers — load_data(), run_inference_for_plot(), compute_nav(), compute_metrics_dict(), sanitize_metrics(), save_json(), NumpyEncoder, metrics_from_ppo_results()
- `exp/exp_1_core.py`: Core PPO experiment (BB+TC+Regret) — trains 3 phases, plots NAV & weights, prints paper comparison
- `exp/exp_2_comparison.py`: 4-method reward comparison — DiffSharpe, EmbDrwdn, Regret, Static 60/40
- `exp/exp_3_ablation.py`: 5-arm ablation — BB on/off × TC on/off × reward type
- `exp/exp_4_sensitivity.py`: Sensitivity analysis — SensitivityTrainer subclass + AlternatingBootstrapCallback; 5 axes × varied values
- `exp/run_all.py`: Master orchestrator — sequential subprocess calls to all 4 experiment scripts

## Experiment Script Design Notes
- Each exp script is standalone: `python exp/exp_N_xxx.py` from project root
- Shared code lives in `exp/utils.py`; imported via `from exp.utils import ...`
- `load_data()` calls `sys.exit(1)` if `data/raw_prices.csv` missing (instructs user to run download)
- All plots saved to `results/` directory (created automatically)
- JSON results use `NumpyEncoder` to handle `np.inf` / `np.nan` → `null`
- `total_timesteps_per_phase=20_480` (10 × n_steps=2048) for fast iteration
- `SensitivityTrainer` (in exp_4) overrides `_build_ppo`, `_evaluate_on_split`, `train_phase`
- `AlternatingBootstrapCallback` (in exp_4) overrides `_on_step` for strict deterministic alternation
- `tests/test_metrics.py`: 12 pytest tests covering annual return, max drawdown, Sharpe, Calmar, Omega, Sortino, edge cases
- `tests/test_oracle.py`: 8 pytest tests covering weight validity, single-best strategy, regret reward, split_end_idx
- `tests/test_bootstrap.py`: 7 pytest tests covering shape, column preservation, NaN-free output, reproducibility, block fracs
- `tests/test_features.py`: 9 pytest tests covering column names, strategy weight formulas, NaN guarantee, date filtering

## Notes on FeatureBuilder Testing
- Use `object.__new__(FeatureBuilder)` and set `fb._prices = synthetic_df` to avoid file I/O in tests
- Synthetic price_df requires columns: `EFA`, `EEM`, `LQD`, `TLT`, `HYG`, `^VIX`, `^TNX`
- After `build()` and `dropna()`, the first valid row index is 60 (pct_change=1 + rolling60std=59 rows dropped)
- `feat_df.index` contains the original integer row positions; use `.loc[feat_df.index]` to align price series

## Known Limitations
1. Bloomberg tickers not available in free APIs → ETF proxies used
2. MOVE index not available → TLT rolling vol proxy used
3. Pre-2001 developed equity data: SPY used as proxy for EFA
4. Paper's exact covariance shrinkage unspecified → regularization with epsilon=1e-6
5. Phase 2/3 checkpoint selection Pareto tie-break: max annual return among Pareto-efficient checkpoints
