---
sidebar_label: Random Search
---

## Random Search

Random search samples hyperparameters uniformly (or from specified distributions). It is a robust baseline and scales trivially with parallelism.

### Pseudocode
```text
for i in 1..N:
  sample params
  evaluate
  record best
```

### Pros / Cons
- Pros: simple, parallel
- Cons: inefficient in high-dim spaces
