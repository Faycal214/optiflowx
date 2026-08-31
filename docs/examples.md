# Worked Examples

The examples are executable documentation. Each script is small enough to run independently and is designed to answer a concrete question.

## Time Series — primary applied examples

| Example | What it demonstrates |
|---|---|
| `examples/08_eviews_time_series_workflow.py` | Workfile creation, EViews-style expressions, equation estimation, ADF and model estimation |
| `examples/09_state_space_kalman.py` | Linear-Gaussian state-space filtering and missing-observation semantics |
| `examples/10_state_space_workflow.py` | End-to-end filtering, smoothing, innovation diagnostics, adequacy and forecasting |

These examples correspond directly to the [Time Series User Guide](time-series/index.md).


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
