# Continuous-time Markov chains and birth-death processes

## CTMC model

A finite homogeneous CTMC is described by a generator matrix $Q=(q_{ij})$ with

$$q_{ii}=-\sum_{j\ne i}q_{ij}, \qquad q_{ij}\ge 0 \text{ for } i\ne j.$$

The transition matrix at time $t$ is

$$P(t)=e^{Qt}.$$

## Creating a CTMC

```python
from stochx.stochastic import ContinuousTimeMarkovChain

Q = [
    [-2.0, 2.0],
    [1.0, -1.0],
]
chain = ContinuousTimeMarkovChain(Q, states=["A", "B"])
```

The constructor validates generator structure before numerical calculations are performed.

## Transition probabilities

The chain can compute transition probabilities from matrix exponentials and the numerical routes supported by the implementation.

```python
P_t = chain.transition_matrix(1.5)
```

The Chapman–Kolmogorov property is

$$P(t+s)=P(t)P(s).$$

## Simulated trajectories

`CTMCPath` stores a simulated path and supports state lookup and occupation statistics.

```python
path = chain.simulate(
    horizon=10.0,
    initial_state="A",
    rng=np.random.default_rng(0),
)
print(path.states)
```

## Birth-death processes

A birth-death process is a nearest-neighbour CTMC on an ordered state space. From state $n$, transitions occur at birth rate $\lambda_n$ and death rate $\mu_n$.

```python
from stochx.stochastic import BirthDeathProcess

process = BirthDeathProcess(
    birth_rates=[1.0, 1.2, 1.4],
    death_rates=[0.5, 0.7, 0.9],
)
```

The implementation validates rate definitions and boundary behavior explicitly.

## When to use which object

- use `MarkovChain` for discrete time;
- use `ContinuousTimeMarkovChain` for finite-state continuous time;
- use `BirthDeathProcess` when transitions are restricted to neighbours;
- use the time-series package when the observations are indexed measurements rather than state-transition trajectories.

## API reference

See the CTMC and birth-death API pages for the exact constructors, transition methods and trajectory operations.
