# Continuous-Time Markov Chains

This page gives a compact mathematical reference for finite homogeneous CTMCs and shows how the concepts map to StochX.

## 1. Transition probabilities

For a homogeneous CTMC,

\[
p_{ij}(t)=P(X_{s+t}=j\mid X_s=i),
\]

and

\[
P(t)=(p_{ij}(t)),
\qquad P(0)=I.
\]

The transition matrices satisfy Chapman–Kolmogorov.

```python
from stochx.stochastic import ContinuousTimeMarkovChain

chain = ContinuousTimeMarkovChain(Q, states=[0, 1, 2])
chain.transition_matrix_at(2.0)
```

## 2. Infinitesimal generator

The generator \(Q=(q_{ij})\) is defined through the infinitesimal behavior

\[
p_{ij}(h)=q_{ij}h+o(h),
\qquad i\ne j.
\]

Off-diagonal rates are non-negative, diagonal rates are non-positive, and every row sums to zero:

\[
\sum_jq_{ij}=0.
\]

```python
chain.generator_matrix
chain.infinitesimal_transition_matrix(h)
```

## 3. Transition matrix and state law

For a finite homogeneous CTMC,

\[
P(t)=e^{tQ},
\]

and for an initial row distribution \(\mu_0\),

\[
\mu_t=\mu_0P(t).
\]

```python
P_t = chain.transition_matrix_at(2.0)
mu_t = chain.state_distribution(mu_0, 2.0)
```

The API also supports a uniformization-based numerical path for transition matrices.

```python
P_t = chain.transition_matrix_at(2.0, method="uniformization")
```

## 4. Chapman–Kolmogorov equations

Homogeneity gives

\[
P(s+t)=P(s)P(t).
\]

```python
chain.chapman_kolmogorov(s, t)
```

## 5. Kolmogorov equations

The forward and backward equations are

\[
\frac{dP(t)}{dt}=P(t)Q,
\]

and

\[
\frac{dP(t)}{dt}=QP(t).
\]

```python
chain.forward_derivative(t)
chain.backward_derivative(t)
```

## 6. Holding times

From state \(i\), the exit rate is

\[
q_i=-q_{ii}.
\]

When \(q_i>0\), the holding time is exponential with parameter \(q_i\).

```python
chain.holding_rate(i)
chain.holding_time(i)
```

If the exit rate is zero, the holding time is infinite in the model represented by the package.

## 7. Embedded jump chain

The states observed at jump times form a discrete-time Markov chain. For \(i\ne j\) and \(q_i>0\),

\[
\widetilde p_{ij}=\frac{q_{ij}}{q_i}
=\frac{q_{ij}}{-q_{ii}}.
\]

```python
chain.jump_chain_matrix()
jump_chain = chain.jump_chain()
```

## 8. Stationary distribution

A distribution \(\pi\) is stationary if

\[
\pi P(t)=\pi,
\qquad \forall t\ge0.
\]

In the finite framework,

\[
\pi Q=0,
\qquad
\sum_i\pi_i=1.
\]

```python
pi = chain.stationary_distribution()
```

## 9. Long-run occupation and cost

The time spent in state \(i\) up to \(T\) is represented by

\[
\int_0^T\mathbf1_{\{X_t=i\}}\,dt.
\]

The corresponding occupation fraction is

\[
\frac1T\int_0^T\mathbf1_{\{X_t=i\}}\,dt.
\]

For a state cost function \(h\), a stationary long-run cost is

\[
\sum_i\pi_i h(i).
\]

```python
path.occupation_time(i, 20.0)
path.occupation_fraction(i, 20.0)
```

## 10. Non-explosion

The explosion time can be represented as

\[
\zeta=\sum_{n\ge1}S_n.
\]

The process is non-explosive when

\[
P(\zeta=+\infty)=1.
\]

For a pure-birth model, this is related to the behavior of sums of reciprocal birth rates.

## 11. Simulation

A jump-by-jump simulation follows these steps:

1. choose the current state;
2. generate the exponential holding time associated with its exit rate;
3. choose the next state according to the jump chain;
4. repeat until the requested time horizon is reached.

```python
path = chain.simulate(t_max=20.0, initial_state=0)
path.times
path.states
path.state_at(5.0)
```

## 12. API map

| Mathematical object | StochX |
|---|---|
| Generator \(Q\) | `ContinuousTimeMarkovChain` |
| Infinitesimal approximation | `infinitesimal_transition_matrix` |
| Transition matrix \(P(t)\) | `transition_matrix_at` |
| State law | `state_distribution` |
| Chapman–Kolmogorov | `chapman_kolmogorov` |
| Forward equation | `forward_derivative` |
| Backward equation | `backward_derivative` |
| Holding rate / time | `holding_rate`, `holding_time` |
| Embedded jump chain | `jump_chain`, `jump_chain_matrix` |
| Stationary law | `stationary_distribution` |
| Simulated path | `CTMCPath` |
| Simulation | `simulate` |
