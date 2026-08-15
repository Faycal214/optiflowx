# MarkovChain

`MarkovChain` represents a finite, homogeneous discrete-time Markov chain with transition matrix

$$P=(p_{ij}), \qquad p_{ij}=\mathbb P(X_{n+1}=j\mid X_n=i).$$

## Constructor

```python
MarkovChain(transition_matrix, states=None, tolerance=1e-12)
```

### Parameters

| Parameter | Description |
|---|---|
| `transition_matrix` | Square row-stochastic matrix $P$. |
| `states` | Optional state labels. Integers are used when omitted. |
| `tolerance` | Numerical tolerance used by validation and classification. |

### Example

```python
import numpy as np
from stochx.stochastic import MarkovChain

P = np.array([[0.8, 0.2], [0.3, 0.7]])
chain = MarkovChain(P, states=["A", "B"])
```

## Properties

| Property | Meaning |
|---|---|
| `states` | Ordered public state labels. |
| `n_states` | Number of states. |
| `transition_matrix` | Validated transition matrix $P$. |
| `communicating_classes` | Communicating classes of the chain. |
| `closed_classes` | Closed communicating classes. |
| `recurrent_states` | States classified as recurrent. |
| `transient_states` | States classified as transient. |
| `is_irreducible` | Whether the chain has one communicating class. |
| `is_aperiodic` | Whether all states have period one. |
| `is_ergodic` | Course-level ergodicity condition exposed by the package. |
| `period` | Period of a state/class under the implementation convention. |

## Methods

### Transition and distribution

- `transition_matrix_at(n)` — compute $P^n$.
- `n_step_transition(n)` — compatibility spelling for $P^n$.
- `state_distribution(initial_distribution, n)` — compute $\mu_0P^n$.
- `chapman_kolmogorov(m, n)` — verify/compute the composition $P^mP^n$.

### Accessibility and classification

- `accessible(i, j)` — test accessibility from state `i` to `j`.
- `communicate(i, j)` — test communication between two states.
- `classify_states()` — classify recurrence/transience.
- `is_absorbing_state(state)` — test whether a state is absorbing.

### First visits and returns

- `first_visit_probability(i, j, n)` — first visit to `j` from `i` at step `n`.
- `first_passage_probability(i, j, n)` — canonical first-passage spelling.
- `first_return_probability(state, n)` — first return at step `n`.
- `return_probability(state)` — return probability.
- `mean_return_time(state)` — mean return time when defined.
- `expected_hitting_time(source, target)` — expected hitting time.
- `hitting_probability(source, target)` — hitting probability.

### Stationarity, limits, and absorption

- `stationary_distribution()` — compute the unique stationary law when the implemented conditions give uniqueness.
- `stationary_distributions()` — compute stationary laws for the general finite-chain case.
- `limiting_distribution(initial_distribution=None)` — compute the limiting law when the course-level conditions apply.
- `absorption_probability(start, target_class)` — absorption probability into a specified closed class.

### Simulation

- `simulate(initial_state, n_steps, rng=None)` — generate a discrete trajectory.

## Complete example

```python
import numpy as np
from stochx.stochastic import MarkovChain

P = np.array([
    [0.8, 0.2],
    [0.3, 0.7],
])
chain = MarkovChain(P, states=["Healthy", "Sick"])

print(chain.transition_matrix_at(5))
print(chain.stationary_distribution())
print(chain.communicating_classes)
```

## Related course material

[Chapter 1 — CMTD](../course_chapitre1.md)
