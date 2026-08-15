# Chapter 2 — Poisson Processes

This page develops the mathematics of Poisson processes and keeps the mathematical discussion separate from the Python API and worked examples.

## 1. Introduction

A Poisson process is introduced as a continuous-time process with non-negative integer values, used to describe the times at which random events occur: calls, customer arrivals, failures, particle emissions, and similar phenomena.

## 2. Counting process

Let \(N(t)\) be the number of occurrences observed in \(]0,t]\). A process \((N(t))_{t\ge0}\) is a counting process if

\[
N(t)\in\mathbb N,
\qquad s<t\Rightarrow N(s)\le N(t),
\]

and if \(N(t)-N(s)\) represents the number of occurrences on \(]s,t]\).

## 3. First definition of the Poisson process

A counting process is a Poisson process with rate \(\lambda>0\) if:

- \(N(0)=0\) almost surely;
- the increments are independent;
- the increments are stationary;
- \(P(N(h)=1)=\lambda h+o(h)\);
- \(P(N(h)\ge2)=o(h)\).

Here \(o(h)/h\to0\) as \(h\to0\).

## 4. Second definition

An equivalent formulation is: \(N(0)=0\), the increments are independent, and for every interval length \(t\),

\[
P(N(t+s)-N(s)=n)=e^{-\lambda t}\frac{(\lambda t)^n}{n!},
\qquad n\in\mathbb N.
\]

This law does not depend on \(s\), which gives stationary increments.

## 5. Equivalence of the definitions

Let

\[
p_n(t)=P(N(t)=n).
\]

The analysis over \([t,t+h]\) gives

\[
p_0'(t)=-\lambda p_0(t),\qquad p_0(0)=1,
\]

and, for \(n\ge1\),

\[
p_n'(t)=-\lambda p_n(t)+\lambda p_{n-1}(t),
\qquad p_n(0)=0.
\]

Solving these equations gives

\[
\boxed{P(N(t)=n)=e^{-\lambda t}\frac{(\lambda t)^n}{n!}}.
\]

Therefore \(N(t)\sim\mathcal P(\lambda t)\) and

\[
N(t+s)-N(s)\sim\mathcal P(\lambda t).
\]

## 6. Inter-occurrence times

If

\[
0=\tau_0<\tau_1<\tau_2<\cdots
\]

are the occurrence times, the inter-arrival times

\[
T_n=\tau_n-\tau_{n-1}
\]

form an i.i.d. sequence with exponential distribution of parameter \(\lambda\):

\[
T_n\sim\mathrm{Exp}(\lambda).
\]

## 7. Conditional occurrence times

Conditionally on \(N(s)=1\), the time of the unique occurrence in \([0,s]\) is uniform:

\[
T_1\mid\{N(s)=1\}\sim\mathcal U([0,s]).
\]

More generally, conditionally on \(N(s)=k\), the \(k\) occurrence times in \([0,s]\) have the distribution of the order statistics of a sample of \(k\) independent uniform random variables on \([0,s]\).

## 8. Superposition

If \(N^{(1)}\) and \(N^{(2)}\) are independent Poisson processes with rates \(\lambda_1\) and \(\lambda_2\), then

\[
N(t)=N^{(1)}(t)+N^{(2)}(t)
\]

is again a Poisson process, with rate

\[
\boxed{\lambda_1+\lambda_2}.
\]

## 9. Splitting / thinning

Starting from a Poisson process of rate \(\lambda\), assign each occurrence independently to type 1 with probability \(p\), or to type 2 with probability \(1-p\). The two resulting processes are independent and have rates

\[
\lambda p
\qquad\text{and}\qquad
\lambda(1-p).
\]

## 10. Non-homogeneous Poisson process

In the non-homogeneous case, the rate depends on time: \(\lambda=\lambda(t)\). Increments are therefore no longer stationary. Introduce the cumulative mean function

\[
\boxed{m(t)=\int_0^t\lambda(u)\,du}.
\]

Locally,

\[
P(\text{one occurrence in }[t,t+h[)=\lambda(t)h+o(h).
\]

This model is suited to phenomena whose occurrence rate varies over time.

## 11. Key results

\[
N(t)\sim\mathcal P(\lambda t),
\qquad
T_n\sim\mathrm{Exp}(\lambda).
\]

Increments over disjoint intervals are independent; in the homogeneous case, their law depends only on the interval lengths. Conditionally on a fixed number of occurrences, the occurrence times are distributed as uniform order statistics.

## 12. Summary

\[
\text{counting}
\rightarrow \text{Poisson definition}
\rightarrow \text{Poisson law}
\rightarrow \text{inter-arrival times}
\]

\[
\rightarrow \text{conditional times}
\rightarrow \text{superposition}
\rightarrow \text{splitting}
\rightarrow \text{non-homogeneous Poisson process}.
\]
