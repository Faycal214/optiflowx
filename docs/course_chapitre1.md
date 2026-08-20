# Chapter 1 — Discrete-Time Markov Chains

## 1. The model

A discrete-time process $(X_n)_{n\ge0}$ on a finite state space $S$ is a Markov chain when

$$\mathbb P(X_{n+1}=j\mid X_n=i,X_{n-1},\ldots,X_0)=\mathbb P(X_{n+1}=j\mid X_n=i).$$

For a homogeneous chain, the transition probabilities do not depend on $n$:

$$p_{ij}=\mathbb P(X_{n+1}=j\mid X_n=i).$$

The matrix $P=(p_{ij})$ is row-stochastic.

## 2. n-step transitions

The Chapman–Kolmogorov relation gives

$$P^{m+n}=P^mP^n.$$

The entry $(P^n)_{ij}$ is the probability of being in state $j$ after $n$ steps when starting from $i$.

StochX exposes this through `transition_matrix_at(n)` and `n_step_transition(n)`.

## 3. Communication structure

A state $j$ is accessible from $i$ if there is an $n$ such that

$$p_{ij}^{(n)}>0.$$

Two states communicate when each is accessible from the other. Communication classes split the state space into regions with different long-run behavior.

Important classifications:

- transient state;
- recurrent state;
- closed class;
- absorbing state;
- irreducible chain;
- periodic or aperiodic chain.

## 4. Stationarity

A probability vector $\pi$ is stationary when

$$\pi P=\pi.$$ 

Under appropriate finite-chain conditions, irreducibility gives a unique stationary distribution. Aperiodicity is then important for convergence from arbitrary initial distributions.

Do not confuse existence of a stationary law with convergence to it.

## 5. First passage and return

The first hitting time of $j$ from $i$ is

$$T_{ij}=\inf\{n\ge1:X_n=j\mid X_0=i\}.$$

The package exposes hitting probabilities, expected hitting times, first-return probabilities and mean return times.

## 6. Absorption

For an absorbing chain, reorder the transition matrix into transient and absorbing blocks and use the fundamental matrix

$$N=(I-Q)^{-1}.$$

This gives expected visits to transient states and, together with the absorbing block, absorption probabilities.

## 7. Simulation

A trajectory is generated from the transition matrix and an initial state/distribution. Pass an explicit NumPy random generator for reproducibility.

```python
import numpy as np
from stochx.stochastic import MarkovChain

chain = MarkovChain([[0.8, 0.2], [0.3, 0.7]], states=["A", "B"])
path = chain.simulate(1000, initial_state="A", rng=np.random.default_rng(0))
```

## 8. Implementation workflow

The StochX implementation turns these definitions into validated objects. Invalid transition matrices fail at construction; state labels remain explicit; transition, classification, hitting and simulation operations share the same state-space representation.

See the [Markov-chain user guide](stochastic/markov-chains.md) and [API reference](api/markov_chain.md).
