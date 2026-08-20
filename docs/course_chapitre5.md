# Chapter 5 — Discrete-Time Martingales

## 1. Filtrations and information

A filtration $(\mathcal F_n)$ is an increasing sequence of sigma-algebras describing the information available through time:

$$\mathcal F_0\subseteq\mathcal F_1\subseteq\cdots.$$

A process adapted to $(\mathcal F_n)$ has $X_n$ measurable with respect to $\mathcal F_n$.

## 2. Martingale definition

A process $(M_n)$ is a martingale if

$$\mathbb E[|M_n|]<\infty$$

and

$$\mathbb E[M_{n+1}\mid\mathcal F_n]=M_n.$$

A submartingale replaces equality by $\ge$, and a supermartingale by $\le$.

## 3. How to verify a martingale

A course-level verification should check:

1. adaptedness;
2. integrability;
3. conditional expectation of the next value.

A process can have constant ordinary expectation without being a martingale. The conditional relation is the essential condition.

## 4. Stopping times

A random time $T$ is a stopping time with respect to $(\mathcal F_n)$ when

$$\{T\le n\}\in\mathcal F_n.$$

The definition ensures that the decision to stop at time $n$ uses only information available by time $n$.

## 5. Stopped processes

Given a process $X_n$ and a stopping time $T$, define

$$X^T_n=X_{n\wedge T}.$$

The stopped process freezes the trajectory after the stopping time.

## 6. Optional-stopping intuition

Optional-stopping results relate the martingale property to stopped expectations, but the theorem requires hypotheses. Integrability, bounded stopping times or other regularity assumptions cannot simply be ignored.

StochX's finite/discrete objects are therefore best used as executable demonstrations and checks of explicit assumptions, not as a replacement for a proof of a theorem.

## 7. Implementation map

- `Filtration` represents the information structure.
- `Martingale` represents the process and its martingale condition.
- `StoppingTime` represents the stopping rule.
- `StoppedProcess` represents the stopped trajectory.

See the [martingale guide](stochastic/martingales.md) and API reference for the Python contract.
