---
sidebar_label: Tree-structured Parzen Estimator (TPE)
---

## Tree-structured Parzen Estimator (TPE)

TPE models P(x|y) using KDEs for good and bad trials and proposes candidates that maximize l(x)/g(x). It's effective for conditional and mixed search spaces.

### Components
- KDE-based densities l(x) and g(x)
- Threshold quantile γ separating good vs bad

### Pros / Cons
- Pros: handles conditional spaces, flexible
- Cons: KDE overhead on large histories
