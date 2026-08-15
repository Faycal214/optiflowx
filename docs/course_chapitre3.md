# Chapter 3 — Continuous-Time Markov Chains (CTMC)

This page develops the mathematics of continuous-time Markov chains and birth-death processes. The mathematical discussion remains separate from the Python API and worked examples.

## 1. Definition of a CTMC

On a probability space, a process \(X=(X_t)_{t\ge0}\) with finite or countable state space is a continuous-time Markov chain when, for times

\[
0=t_0<t_1<\cdots<t_n<t_{n+1},
\]

\[
P(X_{t_{n+1}}=j_{n+1}\mid X_{t_n}=j_n,\ldots,X_{t_0}=j_0)
=P(X_{t_{n+1}}=j_{n+1}\mid X_{t_n}=j_n).
\]

In the homogeneous case,

\[
P(X_{s+t}=j\mid X_s=i)=p_{ij}(t),
\]

so the transition depends only on the elapsed time \(t\). We write

\[
P(t)=(p_{ij}(t)),
\qquad p_{ij}(0)=\delta_{ij}.
\]

Each row of \(P(t)\) is a probability distribution.

## 2. Infinitesimal generator

The generator \(Q=(q_{ij})\) describes behavior over an infinitesimal interval \(h\). For \(i\ne j\),

\[
q_{ij}=\lim_{h\to0}\frac{p_{ij}(h)}{h},
\]

and

\[
p_{ij}(h)=q_{ij}h+o(h),
\qquad
p_{ii}(h)=1+q_{ii}h+o(h).
\]

The rows of \(Q\) sum to zero:

\[
\sum_jq_{ij}=0,
\qquad
q_{ii}=-\sum_{j\ne i}q_{ij}.
\]

The probability of making two or more transitions over an infinitesimal interval is of order \(o(h)\).

## 3. Poisson process as a CTMC

For a Poisson process with rate \(\lambda\), transitions are possible from \(i\) to \(i+1\) with rate \(\lambda\), while the diagonal of \(Q\) compensates for the total exit rate. Thus the Poisson process is an elementary example of a CTMC.

## 4. Kolmogorov equations

The generator governs the evolution of the transition matrix. In the finite matrix case, the backward equation is

\[
P'(t)=QP(t),
\qquad P(0)=I,
\]

which gives

\[
\boxed{P(t)=e^{tQ}}.
\]

The matrix exponential is defined by

\[
e^{tQ}=I+tQ+\frac{t^2Q^2}{2!}+\cdots
=\sum_{k=0}^{\infty}\frac{t^kQ^k}{k!}.
\]

In the general case, especially for an infinite state space, an explicit expression for \(P(t)\) may not be available.

## 5. State distribution

If \(\mu_0\) is the initial law written as a row vector,

\[
\boxed{\mu_t=\mu_0P(t)}.
\]

This is the continuous-time analogue of \(\mu_n=\mu_0P^n\) for discrete-time chains.

## 6. Stationary distribution

A distribution \(\pi\) is stationary if

\[
\pi P(t)=\pi,
\qquad \forall t\ge0.
\]

The generator characterization is

\[
\boxed{\pi Q=0},
\qquad
\pi_i\ge0,
\qquad
\sum_i\pi_i=1.
\]

This equation can be used to find a stationary distribution without explicitly computing the complete matrix \(P(t)\).

## 7. Holding times

The process remains in its current state for a random amount of time before jumping to a new state. The successive holding times describe the path between jump times.

The total exit rate from state \(i\) is

\[
-q_{ii}=\sum_{j\ne i}q_{ij}.
\]

## 8. Embedded jump chain

The states observed at jump times form a discrete-time Markov chain, called the embedded jump chain. When \(-q_{ii}>0\), the probability that the next jump from \(i\) goes to \(j\ne i\) is

\[
r_{ij}=\frac{q_{ij}}{-q_{ii}}.
\]

This construction connects CTMC properties with DTMC properties.

## 9. Return, occupation, and asymptotic behavior

The chapter studies return times and the proportions of time spent in states. If each unit of time spent in state \(i\) generates a cost \(h(i)\), then under a stationary distribution \(\pi\), the mean cost is

\[
\sum_i\pi_i h(i).
\]

For an irreducible, non-explosive, positive-recurrent CTMC, there is a unique stationary distribution \(\pi\) satisfying

\[
\pi Q=0,
\]

and

\[
\lim_{t\to\infty}p_{ij}(t)=\pi_j.
\]

The mean return time is also related to the stationary distribution through a formula involving the exit rate \(-q_{ii}\) and \(\pi_i\).

Non-explosion means that infinitely many jumps do not occur in finite time.

# 10. Birth-death processes

A birth-death process is a special CTMC in which the only possible jumps are

\[
i\to i+1 \quad\text{(birth)},
\]

and, for \(i>0\),

\[
i\to i-1 \quad\text{(death)}.
\]

Introduce birth rates \(\lambda_i\) and death rates \(\mu_i\). The generator is tridiagonal and its transition diagram is labeled by these rates.

## 10.1. Kolmogorov equations

Let

\[
p_k(t)=P(X_t=k).
\]

The Kolmogorov equations describe the balance between incoming and outgoing probability flux for each state \(k\).

## 10.2. Stationary distribution

When a stationary distribution exists, the stationary probabilities are related to successive rates by a recursive relation of the form

\[
\pi_n\propto\prod_{k=0}^{n-1}\frac{\lambda_k}{\mu_{k+1}},
\]

followed by normalization when the sum of the masses is finite.

## 10.3. Linear rates

A common model is

\[
\lambda_n=n\lambda+\alpha,
\]

where \(\alpha\ge0\) represents immigration and \(\lambda\ge0\) is the birth rate per individual. The death rate is proportional to the population size: with \(n\) individuals, a death during \([t,t+h[\) has probability of order \(n\mu h\).

## 10.4. Non-explosion

Let \(\zeta\) be the explosion time. For a pure-birth process, one criterion is

\[
\sum_k\frac1{\lambda_k}=\infty
\quad\Longrightarrow\quad
P(\zeta=\infty)=1.
\]

## 11. Summary

\[
\text{CTMC}
\rightarrow P(t)
\rightarrow Q
\rightarrow\text{Kolmogorov}
\rightarrow\mu_t
\rightarrow\text{stationarity}
\rightarrow\text{holding times}
\rightarrow\text{embedded jump chain}
\rightarrow\text{long-term behavior}
\]

then

\[
\text{birth-death}
\rightarrow\text{generator}
\rightarrow\text{Kolmogorov}
\rightarrow\text{stationarity}
\rightarrow\text{special rates}
\rightarrow\text{non-explosion}.
\]
