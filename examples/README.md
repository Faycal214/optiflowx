# Examples

This folder contains example scripts that run the library's optimizers on
small synthetic datasets. The examples are split so you can run classification
and regression scenarios independently.

Files
- `classification_examples.py`: Runs all optimizers on synthetic
  classification datasets (sklearn metrics and a simple custom metric).
- `regression_examples.py`: Runs all optimizers on synthetic regression
  datasets. Error-based metrics (MSE, RMSE, MAE) are negated via
  `optiflowx.core.metrics.get_metric` so the optimizers always *maximize*
  the returned score (i.e., minimize the underlying error).
- `run_all_optimizers.py`: Backwards-compatible wrapper that runs both.

Quick usage

Run classification examples:
```bash
python examples/classification_examples.py
```

Run regression examples:
```bash
python examples/regression_examples.py
```

Shorten runs for CI/tests

To make examples run faster (for CI or quick checks), set the
`EXAMPLES_MAX_ITERS` environment variable to a small integer. Example:

```bash
EXAMPLES_MAX_ITERS=1 python examples/regression_examples.py
```

This variable controls the `max_iters` passed to each optimizer in the
examples and defaults to `3` if not set.
