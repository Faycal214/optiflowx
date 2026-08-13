# OptiFlowX architecture

OptiFlowX is designed as a **mathematical library with an executable course layer**.

The central rule is:

> **The PDF defines the mathematical object. The class represents the object. The documentation explains the mapping.**

This avoids two common problems in educational scientific libraries: code that contains undocumented formulas, and documentation that explains an API without explaining the mathematics behind it.

## 1. Three layers

### Layer A — course mathematics

Every public stochastic component begins with a notion from one of the five USTHB MSPRO PDFs:

| Chapter | Mathematical domain | Main objects |
|---|---|---|
| 1 | CMTD | transition matrix, transition probabilities, communication classes, recurrence, period, stationary and limiting distributions |
| 2 | Poisson processes | counting process, increments, arrival/interarrival times, conditioning, superposition, thinning |
| 3 | CMTC / birth-death | generator, transition matrix, Kolmogorov equations, holding times, jump chain, stationary law, occupation times |
| 4 | Conditional expectation | conditional expectation, conditional probability, tower property, conditional variance/covariance, projection interpretation |
| 5 | Discrete-time martingales | filtration, adaptedness, martingale/submartingale/supermartingale, stopping times, stopped processes, Doob martingale |

### Layer B — computational model

The class should represent the mathematical object directly.

Examples:

```python
MarkovChain(P)
ContinuousTimeMarkovChain(Q)
PoissonProcess(rate)
BirthDeathProcess(birth_rates, death_rates)
FiniteProbabilitySpace(outcomes, probabilities)
Filtration(partitions)
Martingale(process, filtration)
StoppingTime.from_values(...)
```

Avoid classes whose names describe an implementation trick instead of the mathematical object.

### Layer C — documentation and examples

Documentation should teach the object before teaching the method.

For example, the `stationary_distribution()` page should first state:

$$
\pi P=\pi,
\qquad
\sum_i\pi_i=1,
\qquad
\pi_i\ge 0,
$$

then explain the course result and hypotheses, then show:

```python
pi = chain.stationary_distribution()
```

The method is the last step of the explanation, not the first.

## 2. Professional class design

### 2.1 Module docstring

Each module should state:

- the mathematical chapter it implements;
- the exact scope of the implementation;
- whether the implementation is finite-state, discrete-time, homogeneous, etc.;
- that terminology follows the course material.

### 2.2 Class docstring

Every public class should contain four conceptual blocks:

```text
Mathematical object
-------------------
What is represented mathematically?

Course basis
------------
Which chapter/section introduces the object?

Scope
-----
What assumptions does the implementation make?

Examples
--------
What is the canonical usage?
```

For example, `MarkovChain` explicitly documents the transition matrix and the relation

$$
p_{ij}=P(X_{n+1}=j\mid X_n=i).
$$

### 2.3 Public methods

A public mathematical method should document:

- mathematical meaning;
- parameters and their mathematical role;
- return value;
- hypotheses/validation;
- important course interpretation.

Example:

```python
def stationary_distribution(self) -> np.ndarray:
    """Compute the stationary law pi satisfying pi P = pi.

    Course result
    -------------
    In the irreducible positive-recurrent case, the course gives a unique
    stationary distribution and pi_i = 1 / mu_i.
    """
```

### 2.4 Mathematical comments in code

Comments should explain *why a computation represents the formula*, not restate Python syntax.

Good:

```python
# The course defines stationarity by pi P = pi.
# Solve (P^T - I) pi^T = 0 together with sum(pi)=1.
```

Bad:

```python
# Solve the matrix.
solution = np.linalg.solve(...)
```

### 2.5 Validation

Validation should correspond to mathematical hypotheses.

For example:

- a transition matrix must be stochastic;
- a CTMC generator must have non-negative off-diagonal entries and zero row sums;
- probabilities must be non-negative and sum to one;
- a stopping time must satisfy the filtration measurability condition.

Error messages should say which mathematical condition failed.

## 3. API grouping

Methods should be ordered by mathematical purpose:

1. representation and labels;
2. primary equations/laws;
3. structural properties;
4. asymptotic quantities;
5. simulation;
6. private numerical helpers.

This is the organization used by the refactored `MarkovChain` class.

## 4. Numerical conventions

The package is numerical, not symbolic. Therefore:

- formulas are stated mathematically in documentation/docstrings;
- NumPy/SciPy performs finite-dimensional calculations;
- tolerances are explicit where floating-point comparisons are necessary;
- methods do not silently claim a theorem outside the hypotheses supported by the course.

For example, a limiting-distribution method should refuse an unsupported periodic/reducible case rather than return a misleading numerical guess.

## 5. Documentation contract

Every mathematical concept page should contain:

### Definition

The concept as introduced in the PDF.

### Notation

The symbols used in the chapter, such as `P`, `Q`, `pi`, `mu_n`, `T_i`, `F_n`.

### Result

The theorem/proposition and its hypotheses.

### Interpretation

Only interpretations stated or directly implied by the course material.

### Implementation

Which class and method implement it.

### Example

A small numerical example, preferably corresponding to a course example.

### Limitations

What the finite/numerical API cannot conclude.

## 6. Course-source rule

The five PDFs are the primary mathematical source for the stochastic package.

External mathematical knowledge may be used for implementation details, numerical methods, or software engineering, but it must not silently replace a course definition in the educational documentation.

When the PDF gives a theorem only under hypotheses, those hypotheses must appear on the corresponding documentation page.

When a computational convenience is not itself a course notion, it should be labeled as an **implementation helper**, not presented as a new mathematical concept.
