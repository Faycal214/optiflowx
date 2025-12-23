---
sidebar_label: Grid Search
---

## Grid Search

Grid search exhaustively enumerates combinations from finite sets of hyperparameter values. It is simple but suffers from the curse of dimensionality.

### When to use
- Low-dimensional problems where exhaustive search is feasible

### Pseudocode
```text
for each combination in grid:
  evaluate
  update best
```

### Pros / Cons
- Pros: deterministic, easy to parallelize
- Cons: exponential growth with parameters
