# Discrete-Time Markov Chains

This page gives a compact mathematical reference for finite homogeneous discrete-time Markov chains and shows how the concepts map to OptiFlowX.

## 1. Markov property

Let \((X_n)_{n\in\mathbb N}\) be a process with discrete state space \(S\). The Markov property is

\[
P(X_{n+1}=j\mid X_0=i_0,\ldots,X_n=i)=P(X_{n+1}=j\mid X_n=i).
\]

For a homogeneous chain,

\[
p_{ij}=P(X_{n+1}=j\mid X_n=i).
\]

## 2. Transition matrix

The transition matrix is

\[
P=(p_{ij}),
\qquad p_{ij}\ge0,
\qquad \sum_jp_{ij}=1.
\]

```python
from optiflowx.stochastic import MarkovChain

chain = MarkovChain(
    [[0.7, 0.3],
     [0.4, 0.6]],
    states=["A", "B"],
)
```

## 3. Multi-step transitions

The \(n\)-step transition matrix satisfies

\[
P^{(n)}=P^n,
\]

with the Chapman–Kolmogorov relation

\[
P^{(m+n)}=P^{(m)}P^{(n)}.
\]

```python
P5 = chain.n_step_transition(5)
```

## 4. State distribution

If \(\mu_0\) is the initial row distribution,

\[
\mu_n=\mu_0P^n.
\]

```python
mu_n = chain.state_distribution(mu_0, n=10)
```

## 5. Accessibility and communication

State \(j\) is accessible from \(i\) if there exists \(n\ge0\) such that

\[
p_{ij}^{(n)}>0.
\]

Two states communicate when they are mutually accessible. Communication classes partition the state space. Closed classes cannot be left, and an absorbing state satisfies \(p_{ii}=1\).

```python
chain.accessible("A", "B")
chain.communicate("A", "B")
chain.communicating_classes()
chain.closed_classes()
```

## 6. Recurrence and transience

State \(i\) is recurrent when the probability of eventually returning to \(i\), starting from \(i\), is one; otherwise it is transient. The classical criterion is

\[
\sum_{n=1}^{\infty}p_{ii}^{(n)}=\infty
\]

for recurrence and convergence for transience.

```python
chain.classify_states()
```

## 7. Periodicity

The period is

\[
d(i)=\operatorname{pgcd}\{n\ge1:p_{ii}^{(n)}>0\}.
\]

A state is aperiodic when \(d(i)=1\).

```python
chain.period("A")
chain.is_aperiodic()
```

## 8. Stationary distribution

A distribution \(\pi\) is stationary when

\[
\pi P=\pi,
\qquad \pi_i\ge0,
\qquad \sum_i\pi_i=1.
\]

For an irreducible positive-recurrent chain,

\[
\pi_i=\frac{1}{\mu_i},
\]

where \(\mu_i\) is the mean return time.

```python
pi = chain.stationary_distribution()
```

## 9. Limiting distribution

For an ergodic chain,

\[
\lim_{n\to\infty}p_{ij}^{(n)}=\pi_j.
\]

The implementation rejects unsupported periodic or reducible cases instead of returning an unjustified limit.

```python
limit = chain.limiting_distribution()
```

## 10. Absorption and empirical frequencies

Finite absorbing chains can be analyzed through their closed absorbing classes. For simulated trajectories, empirical state frequencies are available as a separate analysis utility.

```python
prob = chain.absorption_probability("A", {"B"})
```

```python
from optiflowx.stochastic.analysis import empirical_state_frequencies

frequencies = empirical_state_frequencies(path, chain.states)
```

## 11. API map

| Mathematical object | OptiFlowX |
|---|---|
| Transition matrix | `MarkovChain` |
| \(P^n\) | `n_step_transition` |
| State law | `state_distribution` |
| Accessibility | `accessible` |
| Communication | `communicate`, `communicating_classes` |
| Closed classes | `closed_classes` |
| Recurrence / transience | `classify_states` |
| Period | `period` |
| Stationary law | `stationary_distribution` |
| Limiting law | `limiting_distribution` |
| Absorption | `absorption_probability` |
| Simulation | `simulate` |
