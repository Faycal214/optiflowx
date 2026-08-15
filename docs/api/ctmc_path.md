# CTMCPath

`CTMCPath` stores a simulated continuous-time Markov-chain trajectory as jump times and visited states.

## Constructor

```python
CTMCPath(times, states)
```

### Parameters

| Parameter | Description |
|---|---|
| `times` | Jump times of the simulated path. |
| `states` | State labels visited along the path. |

## Properties

| Property | Meaning |
|---|---|
| `times` | Jump-time array. |
| `states` | Tuple of visited state labels. |

## Methods

- `state_at(t)` — return the state occupied at time `t`.
- `occupation_time(state, horizon)` — compute time spent in a state over `[0, horizon]`.
- `occupation_fraction(state, horizon)` — compute the fraction of the horizon spent in a state.

## Example

```python
import numpy as np

from stochx.stochastic import ContinuousTimeMarkovChain

chain = ContinuousTimeMarkovChain([[-2.0, 2.0], [1.0, -1.0]])
path = chain.simulate(5.0, initial_state=0, rng=np.random.default_rng(0))

print(path.state_at(2.0))
print(path.occupation_time(0, 5.0))
print(path.occupation_fraction(0, 5.0))
```

## Related API

[`ContinuousTimeMarkovChain`](continuous_time_markov_chain.md) returns `CTMCPath` objects from its simulation operation.
