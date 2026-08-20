# Simulation

Simulation turns the mathematical model into reproducible sample paths or synthetic series.

## Random generators for time-series models

The time-series namespace includes generators for:

```python
from stochx.timeseries import white_noise, ar, ma, arma, random_walk, sarma

noise = white_noise(100, rng=np.random.default_rng(0))
ar1 = ar(1, [0.6], 100, rng=np.random.default_rng(0))
ma1 = ma(1, [0.4], 100, rng=np.random.default_rng(0))
```

These functions are useful for unit tests, demonstrations and controlled experiments because the random generator can be passed explicitly.

## Stochastic-process simulation

Markov, Poisson, CTMC and birth-death objects expose simulation methods that use the same reproducibility principle:

```python
rng = np.random.default_rng(0)
path = chain.simulate(initial_state="A", n_steps=1000, rng=rng)
```

## Reproducibility rule

For a deterministic experiment:

1. fix the model parameters;
2. fix the input data;
3. create one explicit random generator;
4. pass it to the simulation function;
5. record the package version and relevant numerical settings.

## Simulation versus estimation

A generated series is not a fitted model. Use simulation to understand sampling variability and validate implementation behavior; use estimation workflows for parameter inference from observed data.

## Validation pattern

For a new stochastic model, compare simulation summaries with known theoretical properties:

- mean and variance;
- event counts or occupation times;
- stationary frequencies;
- autocorrelation structure;
- limiting behavior.

This is how simulation becomes a numerical test rather than only a visualization tool.
