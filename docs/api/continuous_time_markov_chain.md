# ContinuousTimeMarkovChain

`ContinuousTimeMarkovChain` represents a finite homogeneous continuous-time Markov chain with generator $Q$.

$$q_{ij}\ge 0\;(i\ne j), \qquad \sum_j q_{ij}=0.$$

The transition matrix is $P(t)=e^{Qt}$.

## Constructor

```python
ContinuousTimeMarkovChain(generator, states=None, tolerance=1e-12)
```

### Parameters

| Parameter | Description |
|---|---|
| `generator` | Square infinitesimal generator $Q$. |
| `states` | Optional state labels. |
| `tolerance` | Numerical tolerance for validation and numerical operations. |

### Properties

| Property | Meaning |
|---|---|
| `states` | Ordered state labels. |
| `n_states` | Number of states. |
| `generator` | Validated generator $Q$. |
| `generator_matrix` | Alias for `generator`. |
| `holding_rates` | State holding rates $-q_{ii}$. |
| `jump_chain_matrix` | Embedded jump-chain transition matrix. |

## Methods

### Transition probabilities

- `transition_matrix(t)` — compute $P(t)=e^{Qt}$.
- `transition_matrix_at(t)` — canonical time-dependent API.
- `transition_matrix_uniformized(t, ...)` — compute $P(t)$ by Jensen uniformization.
- `state_distribution(initial_distribution, t)` — compute $\mu_0P(t)$.
- `chapman_kolmogorov(s, t)` — use the semigroup relation $P(s+t)=P(s)P(t)$.

### Kolmogorov equations

- `forward_derivative(t)` / `forward_equation(t)` — $P(t)Q$.
- `backward_derivative(t)` / `backward_equation(t)` — $QP(t)$.
- `infinitesimal_transition_matrix(h)` — first-order matrix $I+hQ$.

### Holding and jump structure

- `holding_rate(state)` — rate $-q_{ii}$.
- `holding_time(state, rng=None)` — exponential holding time.
- `jump_chain()` — return the embedded `MarkovChain`.
- `communicating_classes()` — communicating classes via the jump chain.

### Stationarity and long-run quantities

- `stationary_distribution()` — stationary law satisfying $\pi Q=0$ when the implemented conditions give a unique law.
- `stationary_distribution_from_jump_chain()` — derive stationarity from the embedded chain when applicable.
- `mean_return_time(state)` — continuous-time return quantity.
- `long_run_cost(costs)` — stationary weighted state cost.

### Simulation

- `simulate(initial_state, t_max, rng=None)` — generate a jump-time trajectory.

## Uniformization

For numerically stable computation, the API exposes the Jensen uniformization representation

$$P(t)=e^{-\nu t}\sum_{k=0}^{\infty}\frac{(\nu t)^k}{k!}R^k,$$

with $R=I+Q/\nu$ and $\nu\ge\max_i(-q_{ii})$.

## Example

```python
import numpy as np
from optiflowx.stochastic import ContinuousTimeMarkovChain

Q = np.array([
    [-2.0, 2.0],
    [1.0, -1.0],
])
ctmc = ContinuousTimeMarkovChain(Q, states=["A", "B"])

P1 = ctmc.transition_matrix_at(2.0)
P2 = ctmc.transition_matrix_uniformized(2.0)
print(np.max(np.abs(P1 - P2)))
```

## CTMCPath

`CTMCPath` represents the simulated piecewise-constant state trajectory.

### Methods

- `state_at(t)` — state occupied at time `t`.
- `occupation_time(state, horizon)` — time spent in a state up to the horizon.
- `occupation_fraction(state, horizon)` — occupation time divided by the horizon.

## Related course material

[Chapter 3 — CMTC](../course_chapitre3.md)
