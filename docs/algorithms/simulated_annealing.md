---
sidebar_label: Simulated Annealing (SA)
---

## Simulated Annealing (SA)

Simulated annealing is a probabilistic single-solution metaheuristic that accepts uphill moves with a probability that decreases over time (the cooling schedule).

### Key ideas
- Neighbor sampling
- Temperature T controls acceptance of worse solutions
- Cooling schedule (linear, exponential, etc.)

### Iteration pseudocode
```text
Let s = s0, T = T0
For k = 1..k_max:
  T = schedule(T0, k)
  s_new = random_neighbor(s)
  If accept(s_new, s, T): s = s_new
Return best found
```

### When to use
- Problems requiring occasional uphill moves to escape local minima

### Pros / Cons
- Pros: simple, can escape local minima
- Cons: needs careful schedule tuning
