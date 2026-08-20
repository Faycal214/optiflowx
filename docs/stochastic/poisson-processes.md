# Poisson processes

## Homogeneous Poisson process

A Poisson process $N(t)$ with rate $\lambda>0$ satisfies independent increments and

$$N(t)-N(s)\sim\operatorname{Poisson}(\lambda(t-s)), \qquad t>s.$$

Inter-arrival times are i.i.d. exponential random variables with rate $\lambda$.

## Creating a process

```python
from stochx.stochastic import PoissonProcess

process = PoissonProcess(rate=2.0)
```

The public object exposes probability, counting and simulation operations according to the process API.

## Simulation

```python
path = process.simulate(
    horizon=10.0,
    rng=np.random.default_rng(0),
)
```

Set the random generator explicitly when reproducibility matters.

## Non-homogeneous Poisson processes

The intensity may vary with time:

$$N(t)-N(s)\sim\operatorname{Poisson}\left(\int_s^t\lambda(u)\,du\right).$$

```python
from stochx.stochastic import NonHomogeneousPoissonProcess

process = NonHomogeneousPoissonProcess(intensity=lambda t: 1.0 + 0.2*t)
```

The intensity function must satisfy the mathematical conditions required by the implementation.

## Practical questions

Use the Poisson API when the random object of interest is a **counting process or arrival process**, not when the observations are a regularly sampled continuous-valued economic series. For the latter, use the time-series guide.

## API reference

See the Poisson API page for constructor parameters, probabilities and simulation methods.
