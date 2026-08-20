# Chapter 3 — Continuous-Time Markov Chains and Birth-Death Processes

## 1. CTMC definition

A finite-state continuous-time Markov chain is described through its transition function

$$p_{ij}(t)=\mathbb P(X_{t+s}=j\mid X_s=i),$$

which satisfies the continuous-time Chapman–Kolmogorov equations.

## 2. Generator matrix

The infinitesimal generator $Q$ contains transition rates:

$$q_{ij}\ge0\quad(i\ne j),$$

$$q_{ii}=-\sum_{j\ne i}q_{ij}.$$

Rows of $Q$ sum to zero. The diagonal entry determines the total holding rate in the state.

## 3. Holding times and jumps

When the chain is in state $i$, the holding time is exponential with rate

$$-q_{ii}.$$

Conditional on a jump occurring, the probability of jumping from $i$ to $j\ne i$ is

$$\frac{q_{ij}}{-q_{ii}}.$$

This gives a direct simulation construction.

## 4. Transition semigroup

For a homogeneous finite CTMC,

$$P(t)=e^{Qt}.$$

It satisfies

$$P(t+s)=P(t)P(s),$$

and $P(0)=I$.

StochX exposes transition probabilities and trajectory objects around this generator representation.

## 5. Birth-death processes

A birth-death process is a CTMC whose state changes only between neighbouring states:

$$n\to n+1\quad\text{at rate }\lambda_n,$$

$$n\to n-1\quad\text{at rate }\mu_n.$$

The model is useful for populations, queues, inventories and simple growth/decay mechanisms.

## 6. Boundary conditions

At a lower or upper boundary, the corresponding rate must be treated explicitly. A death rate at state zero, for example, cannot create a transition to a negative state in a standard birth-death model.

## 7. Paths and occupation times

`CTMCPath` represents the simulated trajectory and supports state lookup and occupation statistics. This keeps path-based questions separate from matrix-level transition calculations.

## 8. StochX objects

- `ContinuousTimeMarkovChain` — generator-based finite CTMC.
- `CTMCPath` — simulated trajectory and occupation information.
- `BirthDeathProcess` — nearest-neighbour CTMC specialization.

See the [CTMC/birth-death guide](stochastic/ctmc-birth-death.md) and API reference.
