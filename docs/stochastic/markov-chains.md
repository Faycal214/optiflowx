# Discrete-time Markov chains

## Mathematical object

A finite homogeneous DTMC is defined by a transition matrix

$$P=(p_{ij}), \qquad p_{ij}=\mathbb P(X_{n+1}=j\mid X_n=i).$$

Every row is a probability vector. The Markov property says that the next state depends on the present state, not the full past.

## Creating a chain

```python
import numpy as np
from stochx.stochastic import MarkovChain

P = np.array([
    [0.8, 0.2],
    [0.3, 0.7],
])

chain = MarkovChain(P, states=["A", "B"])
```

The transition matrix and public state labels are validated when the object is created.

## Main questions

### n-step transitions

```python
chain.transition_matrix_at(5)
```

computes $P^5$.

### State distributions

For initial distribution $\mu_0$:

$$\mu_n=\mu_0P^n.$$

```python
chain.state_distribution([1.0, 0.0], 5)
```

### Communication and classification

```python
chain.accessible("A", "B")
chain.communicate("A", "B")
chain.classify_states()
```

The chain can expose communicating classes, closed classes, recurrent states and transient states.

### Stationarity

A stationary distribution $\pi$ satisfies

$$\pi P=\pi, \qquad \sum_i\pi_i=1.$$

```python
chain.stationary_distribution()
```

For a general finite chain, `stationary_distributions()` can expose the broader solution set.

### Hitting and return questions

Methods such as `hitting_probability`, `expected_hitting_time`, `first_return_probability` and `mean_return_time` make the first-passage theory directly executable.

## Simulation

```python
path = chain.simulate(
    initial_state="A",
    n_steps=10_000,
    rng=np.random.default_rng(0),
)
```

Use an explicit random generator when a simulation must be reproducible.

## What to check

Before using a chain for interpretation, inspect irreducibility, periodicity and closed classes. A stationary distribution can exist without being unique, and a limiting distribution requires stronger conditions.

## API reference

See [MarkovChain API](../api/markov_chain.md) for the complete constructor, properties and methods.
