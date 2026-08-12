# PPO Dynamic Rebalancing — Oracle Regret Reward Reproduction

A faithful reproduction of the paper **"Dynamic Portfolio Rebalancing via PPO with Oracle-Based Negative Sharpe Regret Reward"**, implementing a reinforcement learning agent that dynamically rebalances across three tradable strategy sleeves to outperform a static 60/40 benchmark.

---

## Research Objectives

| # | Objective |
|---|-----------|
| 1 | Reproduce the PPO training pipeline with Oracle-based negative Sharpe regret reward |
| 2 | Verify regret-PPO improves the 60/40 benchmark in out-of-sample annual return |
| 3 | Verify regret-PPO improves max drawdown in at least two of three test phases |
| 4 | Reproduce sliding-window phase training with transfer learning |
| 5 | Compare four reward formulations: static 60/40, Differential Sharpe, Embedded Drawdown, Oracle Regret |
| 6 | Ablate bootstrap augmentation and TC scheduling contributions |
| 7 | Quantify sensitivity to paper ambiguities (schedule, block length, GAE λ, PPO epochs) |

---

## Repository Structure

```
.
├── data/
│   ├── download.py       # Massive API + yfinance data pipeline
│   └── raw_prices.csv    # Downloaded price data (generated)
├── src/
│   ├── environment.py    # Custom Gymnasium portfolio environment (19-dim obs, 3-dim action)
│   ├── features.py       # Feature engineering: strategy returns, rolling stats
│   ├── oracle.py         # Oracle optimizer: future-looking Sharpe-minus-TC
│   ├── bootstrap.py      # Circular block bootstrap for data augmentation
│   ├── scheduler.py      # Transaction-cost scheduler (power-law ramp)
│   ├── trainer.py        # PhaseTrainer: 3-phase PPO with transfer learning
│   ├── metrics.py        # Performance metrics: CAGR, MDD, Sharpe, Calmar, Omega, Sortino
│   └── visualization.py  # Plotting utilities
├── exp/
│   ├── utils.py          # Shared experiment utilities
│   ├── exp_1_core.py     # Core PPO (BB + TC + Regret) vs 60/40 benchmark
│   ├── exp_2_comparison.py # 4-method reward comparison
│   ├── exp_3_ablation.py # 5-arm ablation study
│   ├── exp_4_sensitivity.py # Sensitivity analysis on paper ambiguities
│   └── run_all.py        # Master orchestrator
├── tests/
│   ├── test_environment.py
│   ├── test_trainer.py
│   ├── test_features.py
│   ├── test_oracle.py
│   ├── test_bootstrap.py
│   └── test_metrics.py
├── results/
│   └── RESULTS.md        # Experiment results summary
├── AGENTS.md             # Persistent implementation knowledge
└── README.md
```

---

## Strategy Sleeves

| Strategy | Composition |
|----------|------------|
| Strategy 1 | 100% Developed Equities (EFA proxy) |
| Strategy 2 (60/40) | 55% EFA + 5% EEM + 20% LQD + 20% TLT |
| Strategy 3 | 100% Global Govies (TLT proxy) |

---

## Phase Windows

| Phase | Train | Validation | Test |
|-------|-------|------------|------|
| 1 | 1996-02-01 → 2012-01-01 | 2012-01-01 → 2015-01-01 | 2015-01-01 → 2020-01-01 |
| 2 | 2002-01-01 → 2016-01-01 | 2016-01-01 → 2020-01-01 | 2020-01-01 → 2022-01-01 |
| 3 | 2009-01-01 → 2018-01-01 | 2018-01-01 → 2022-01-01 | 2022-01-01 → 2024-01-01 |

---

## Method

### Observation Space (19-dimensional)
```
[0-2]   strat_1, strat_2, strat_3   current daily returns
[3-5]   hy_spread, vix_chg, move_proxy   contextual changes
[6-8]   mean40_s1/s2/s3   40-day rolling means of strategy returns
[9-11]  std60_s1/s2/s3    60-day rolling stds of strategy returns
[12-14] std60_ctx1/2/3    60-day rolling stds of contextual series
[15-17] prev_w1, prev_w2, prev_w3   previous portfolio weights
[18]    tc_scalar   current training transaction-cost schedule value
```

### Oracle Reward
During training only, an Oracle allocation `w*` is computed by solving:
```
argmax_w  [ w'μ̄ₜ - Rƒ ] / √(w'Σ̄ₜw)  −  TC_train(t) · ‖w − w_{t-1}‖₁
```
subject to `wᵢ ≥ 0`, `Σwᵢ = 1`, with `n = 14` business-day look-ahead.

Reward: `Rₜ = −μ̄ₜ · (w* − wₜ)'`

### PPO Hyperparameters
| Parameter | Value |
|-----------|-------|
| γ (discount) | 0.99 |
| Learning rate | 0.001 |
| Clip range | 0.2 |
| n_steps | 2048 |
| Batch size | 64 |
| VF coefficient | 1.0 |
| Entropy start | 5e-5 (decayed to 0 over 10% of training) |
| GAE λ | 0.95 *(paper omits — documented assumption)* |
| Network | 19→64→64→3/1 with Tanh |

---

## Data Sources

Primary: **Massive REST API** (`MASSIVE_TOKEN` environment variable required)

Fallback: **yfinance** (used automatically if Massive key is absent or coverage is incomplete)

> **Note**: The paper uses Bloomberg institutional indices not available in public APIs.
> ETF proxies are used (see AGENTS.md for full mapping table).

---

## Installation

```bash
pip install stable-baselines3 gymnasium torch cvxpy scipy pandas numpy \
            quantstats yfinance matplotlib seaborn
```

Or install from a `requirements.txt` if provided:
```bash
pip install -r requirements.txt
```

---

## Running the Experiments

```bash
# Step 1: Download market data (Massive → yfinance fallback)
export MASSIVE_TOKEN=your_api_key_here   # optional
python data/download.py

# Step 2: Run individual experiments
python exp/exp_1_core.py          # ~3 min per phase on CPU
python exp/exp_2_comparison.py    # ~12 min total
python exp/exp_3_ablation.py      # ~15 min total
python exp/exp_4_sensitivity.py   # ~20 min total

# Or run everything sequentially
python exp/run_all.py

# Step 3: Run tests
pytest tests/ -v
```

Results are saved to `results/` as JSON metrics files and PNG plots.

---

## Expected Results

Paper quantitative anchors (test annual return):

| Phase | 60/40 benchmark | Regret-PPO (paper) |
|-------|----------------|-------------------|
| 1 | +5.6% | +6.4% |
| 2 | +10.5% | +12.8% |
| 3 | −2.6% | −0.7% |

> Exact replication may differ due to Bloomberg data unavailability, ETF proxy
> limitations (shorter history), and unspecified paper hyperparameters.
> See `results/RESULTS.md` for actual reproduced values.

---

## Documented Implementation Assumptions

1. **GAE lambda**: 0.95 (SB3 default; paper omits)
2. **PPO epochs per update**: 10 (SB3 default; paper omits)
3. **Bootstrap schedule**: Algorithm 1 (Bernoulli 0.7 per 10 episodes)
4. **Bootstrap block length**: 80% of training period
5. **Covariance regularization**: ε = 1e-6 (for near-singular matrices)
6. **Inference mode**: Deterministic (mean action)
7. **Pareto checkpoint selection**: max annual return among Pareto-efficient models
8. **Evaluation TC formula**: R_net = w'μ − 0.0025 · ‖w − w_{t-1}‖₁

---

## Citation

If reproducing from a specific paper, please cite the original work and note that
this is an independent implementation using public data proxies.
