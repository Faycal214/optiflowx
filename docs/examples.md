# Worked Examples

The examples are executable documentation. Each script is small enough to run independently and is designed to answer a concrete question.

## Time Series — primary applied examples

| Example | What it demonstrates |
|---|---|
| `examples/08_eviews_time_series_workflow.py` | Workfile creation, EViews-style expressions, equation estimation, ADF and model estimation |
| `examples/09_state_space_kalman.py` | Linear-Gaussian state-space filtering and missing-observation semantics |
| `examples/10_state_space_workflow.py` | End-to-end filtering, smoothing, innovation diagnostics, adequacy and forecasting |

These examples correspond directly to the [Time Series User Guide](time-series/index.md).

## Stochastic Processes

| Example | What it demonstrates |
|---|---|
| `examples/01_discrete_markov_chain.py` | Markov-chain construction, transition probabilities and stationarity |
| `examples/02_poisson_process.py` | Homogeneous Poisson-process construction |
| `examples/02_poisson_complete.py` | Extended Poisson and non-homogeneous operations |
| `examples/03_continuous_markov_chain.py` | Core CTMC workflow |
| `examples/03_cmtc_complete.py` | CTMC transitions, paths and holding/jump operations |
| `examples/04_birth_death_process.py` | Birth-death construction and formulas |
| `examples/05_conditional_expectation.py` | Finite conditional expectation |
| `examples/05_conditional_expectation_complete.py` | Extended probability-space operations |
| `examples/06_martingale.py` | Discrete-time martingale workflow |
| `examples/07_api_operations.py` | Cross-namespace public API operations |
| `examples/api_quickstart.py` | Minimal package entry point |

## How to use an example

Run a single example:

```bash
python examples/08_eviews_time_series_workflow.py
```

Or run the full example suite:

```bash
for example in examples/*.py; do
    echo "=== $example ==="
    python "$example"
done
```

The CI pipeline executes the same scripts. This keeps the documentation examples synchronized with the actual public API instead of allowing tutorial code to become stale.
