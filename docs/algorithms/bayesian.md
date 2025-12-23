---
sidebar_label: Bayesian Optimization
---

## Bayesian Optimization

Bayesian optimization is a model-based strategy for optimizing expensive black-box functions using a surrogate (e.g., Gaussian Process) and an acquisition function.

### Workflow (concise)
1. Fit surrogate to observed data
2. Maximize acquisition to propose next point
3. Evaluate objective
4. Update surrogate and repeat

### Typical components
- Surrogate model (GP, Random Forest, etc.)
- Acquisition (EI, PI, UCB)

### Pros / Cons
- Pros: sample-efficient for expensive evaluations
- Cons: surrogate overhead and scaling limits
