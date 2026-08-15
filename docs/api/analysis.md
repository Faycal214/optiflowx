# Analysis utilities

## empirical_state_frequencies

```python
empirical_state_frequencies(path, states)
```

Compute empirical visit frequencies of states along a discrete trajectory.

### Parameters

| Parameter | Description |
|---|---|
| `path` | Sequence of observed state labels. |
| `states` | State labels defining the output order. |

### Returns

A NumPy array containing one empirical frequency per supplied state, in the supplied order.

### Example

```python
from stochx.stochastic.analysis import empirical_state_frequencies

path = ["A", "B", "A", "A", "B"]
frequencies = empirical_state_frequencies(path, ["A", "B"])
print(frequencies)
```

This utility is intentionally standalone because it analyzes a trajectory without requiring ownership by a particular stochastic-process object.
