# Chapter 1 — Discrete-Time Markov Chains (DTMC)

This page presents the mathematical development of discrete-time Markov chains and keeps the mathematical discussion separate from the Python API and worked examples.

## 1. Stochastic processes

A stochastic process is a family of random variables defined on the same probability space and indexed by a parameter, usually time. It can be viewed as a mapping

\[
X:\Omega\times T\longrightarrow E,
\qquad (\omega,t)\longmapsto X_t(\omega),
\]

with a random variable \(X_t\) for every instant \(t\). Processes may be classified according to whether time and the state space are discrete or continuous. A DTMC corresponds to discrete time and a discrete state space.

## 2. Homogeneous Markov chain

Let \((X_n)_{n\in\mathbb N}\) be a discrete-time process with discrete state space \(S\). The Markov property is

\[
P(X_{n+1}=j\mid X_0=i_0,\ldots,X_n=i)=P(X_{n+1}=j\mid X_n=i).
\]

The chain is homogeneous when the transition probabilities do not depend on time:

\[
P(X_{n+1}=j\mid X_n=i)=P(X_1=j\mid X_0=i).
\]

We write

\[
p_{ij}=P(X_{n+1}=j\mid X_n=i).
\]

## 3. Transition matrix and associated graph

The probabilities are collected in

\[
P=(p_{ij})_{i,j\in S},
\]

with

\[
p_{ij}\ge0,
\qquad \sum_{j\in S}p_{ij}=1.
\]

The associated graph is directed: states are vertices and a possible transition is represented by an edge weighted by its probability.

### Random walk on \(\mathbb Z\)

A standard form is

\[
p_{ij}=\begin{cases}
p,&j=i+1,\\
q,&j=i-1,\\
r,&j=i,\\
0,&\text{otherwise},
\end{cases}
\qquad p+q+r=1.
\]

## 4. Initial law and characterization

The initial law is

\[
\mu_0=(P(X_0=i))_{i\in S}.
\]

A homogeneous chain is fully characterized by \(P\) and \(\mu_0\). For a path \(i_0,\ldots,i_n\),

\[
P(X_0=i_0,\ldots,X_n=i_n)
=\mu_0(i_0)p_{i_0i_1}\cdots p_{i_{n-1}i_n}.
\]

## 5. Multi-step transitions

The probability of going from \(i\) to \(j\) in \(n\) transitions is

\[
p_{ij}^{(n)}=P(X_{m+n}=j\mid X_m=i),
\]

and

\[
P^{(n)}=(p_{ij}^{(n)})_{i,j\in S}.
\]

The matrix \(P^{(n)}\) is stochastic.

## 6. Chapman–Kolmogorov equations

For \(m,n\ge0\),

\[
p_{ij}^{(n+m)}=\sum_{k\in S}p_{ik}^{(m)}p_{kj}^{(n)}.
\]

In matrix form,

\[
P^{(m+n)}=P^{(m)}P^{(n)}.
\]

Since \(P^{(1)}=P\),

\[
P^{(n)}=P^n.
\]

The element \(p_{ij}^{(n)}\) must be distinguished from the scalar power \((p_{ij})^n\).

## 7. Recursive construction

If \((\xi_n)\) is an i.i.d. sequence, if \(X_0\) is independent of that sequence, and if

\[
X_{n+1}=f(X_n,\xi_{n+1}),
\]

then, under the assumptions of the corresponding result, the process constructed this way is a homogeneous Markov chain.

## 8. State distribution

Let

\[
\mu_n=(P(X_n=i))_{i\in S}.
\]

The total-probability formula gives

\[
\mu_n=\mu_0P^n.
\]

## 9. First visit

The probability of a first visit to \(j\) from \(i\) at time \(n\) is

\[
f_{ij}^{(n)}=P(X_n=j,X_{n-1}\ne j,\ldots,X_1\ne j\mid X_0=i).
\]

The total probability of ever visiting \(j\) is

\[
f_{ij}=\sum_{n=1}^{\infty}f_{ij}^{(n)}.
\]

For \(i=j\), \(f_{ii}^{(n)}\) is a first-return probability.

## 10. Accessibility, communication, and classes

State \(j\) is accessible from \(i\) if there exists \(n\ge0\) such that

\[
p_{ij}^{(n)}>0.
\]

Two states communicate when each is accessible from the other. Communication is an equivalence relation; its classes form a partition of \(S\).

A chain is **irreducible** when it has only one communication class. A class is **closed** when it is impossible to leave it. A state is **absorbing** if

\[
p_{ii}=1.
\]

## 11. Recurrence and transience

State \(j\) is recurrent if

\[
f_{jj}=1,
\]

and transient if \(f_{jj}<1\).

The number of returns to \(i\), starting from \(i\), is associated with

\[
N(i,i)=\sum_{n=1}^{\infty}\mathbf 1_{\{X_n=i\}}\mathbf 1_{\{X_0=i\}}.
\]

The expected number of returns satisfies

\[
E[N(i,i)]=\sum_{n=1}^{\infty}p_{ii}^{(n)}.
\]

Thus,

\[
i\text{ recurrent}\iff\sum_{n=1}^{\infty}p_{ii}^{(n)}=\infty,
\]

while convergence of the series characterizes transience. Recurrence is a class property.

## 12. Hitting time and mean return time

The first hitting time of \(j\), starting from \(i\), is

\[
T_{ij}=\min\{n\ge1:X_n=j\mid X_0=i\}.
\]

Then

\[
f_{ij}^{(n)}=P(T_{ij}=n),
\]

and

\[
\mu_{ij}=E(T_{ij}\mid X_0=i)=\sum_{n=1}^{\infty}nf_{ij}^{(n)}.
\]

The mean return time to \(j\) is

\[
\mu_j=E(T_{jj}\mid X_0=j).
\]

The multi-step transition probabilities satisfy the decomposition

\[
p_{ij}^{(n)}=\sum_{k=1}^{n}f_{ij}^{(k)}p_{jj}^{(n-k)}.
\]

## 13. Null and positive recurrence

A recurrent state is **positive recurrent** if

\[
\mu_j<\infty,
\]

and **null recurrent** if

\[
\mu_j=\infty.
\]

The three categories can also be characterized using the series \(\sum p_{ii}^{(n)}\) and the limit of \(p_{ii}^{(n)}\): transience when the series converges; null recurrence when the series diverges but \(p_{ii}^{(n)}\to0\); and positive recurrence when the series diverges and the limit is strictly positive.

## 14. Periodicity and ergodicity

The period of a state is

\[
d(i)=\operatorname{pgcd}\{n\ge1:p_{ii}^{(n)}>0\}.
\]

A state is aperiodic when \(d(i)=1\). Periodicity is a class property.

A state is **ergodic** when it is positive recurrent and aperiodic. A chain is ergodic when all its states are ergodic. In a finite state space, an irreducible chain is positive recurrent and, if it is also aperiodic, it is ergodic.

## 15. Stationary distribution

A distribution \(\pi\) is stationary if

\[
\pi=\pi P,
\qquad \pi_j\ge0,
\qquad \sum_{j\in S}\pi_j=1.
\]

State by state,

\[
\pi_j=\sum_{i\in S}\pi_i p_{ij}.
\]

A stationary distribution therefore also satisfies \(\pi=\pi P^n\) for every \(n\ge1\).

Several cases arise: uniqueness for some irreducible chains, multiple distributions when several closed classes exist, and absence of a stationary distribution for some transient models.

If a stationary distribution exists, it assigns zero mass to transient or null-recurrent states.

If the chain is irreducible and positive recurrent, there is a unique stationary distribution and

\[
\boxed{\pi_j=\frac1{\mu_j}}.
\]

The values \(\pi_j\) represent the long-run proportion of time spent in each state.

## 16. Limiting distribution

The law of \(X_n\) is

\[
\mu_n=\mu_0P^n.
\]

One then studies the conditions under which the limit \(\lim_{n\to\infty}\mu_n\) exists. For an ergodic chain, \(P^n\) converges to a matrix whose rows are identical, and the limiting distribution coincides with the stationary distribution.

## 17. Absorbing chains

The absorbing case is obtained when some classes are absorbing. Reordering the states makes it possible to separate transient states from absorbing states and study absorption probabilities and absorption times through the corresponding matrix blocks.

## 18. Summary

\[
\text{DTMC}
\rightarrow \text{transition}
\rightarrow \text{Chapman–Kolmogorov}
\rightarrow \text{state law}
\rightarrow \text{first visits}
\]

\[
\rightarrow \text{communication}
\rightarrow \text{recurrence/transience}
\rightarrow \text{return times}
\rightarrow \text{periodicity}
\rightarrow \text{stationarity}
\rightarrow \text{limiting distribution/absorption}.
\] 
