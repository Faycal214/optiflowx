# Chapter 4 — Conditional Expectation on Finite Spaces

## 1. The idea

Conditional expectation is the mathematical operation that replaces a random variable by the best information-constrained average available under a smaller information set.

For a finite partition $\mathcal G=\{A_1,\ldots,A_m\}$, $\mathbb E[X\mid\mathcal G]$ is constant on each cell and satisfies

$$\sum_{\omega\in A_k}\mathbb E[X\mid\mathcal G](\omega)\,\mathbb P(\omega)=\sum_{\omega\in A_k}X(\omega)\mathbb P(\omega).$$

Equivalently, when $\mathbb P(A_k)>0$,

$$\mathbb E[X\mid\mathcal G](\omega)=\frac{\mathbb E[X\mathbf 1_{A_k}]}{\mathbb P(A_k)},\qquad \omega\in A_k.$$

## 2. Finite probability spaces

The implementation begins with explicit outcomes and a validated probability vector. This removes ambiguity about the underlying measurable space in course exercises.

## 3. Random variables

A finite random variable is a mapping from outcomes to values. Its expectation is

$$\mathbb E[X]=\sum_{\omega}X(\omega)P(\omega).$$

Variance follows from

$$\operatorname{Var}(X)=\mathbb E[X^2]-\mathbb E[X]^2.$$

## 4. Partitions as information

A partition groups outcomes that cannot be distinguished with the current information. Conditional expectation is therefore a function that is constant on each cell.

## 5. Fundamental properties

The course implementation should preserve the classical properties:

- linearity;
- positivity;
- monotonicity;
- tower property;
- pull-out property for measurable variables;
- preservation of constants;
- conditional Jensen inequality in settings where it is applicable.

## 6. Tower property

For nested information, one obtains

$$\mathbb E[\mathbb E[X\mid\mathcal G]\mid\mathcal H]=\mathbb E[X\mid\mathcal H],\qquad \mathcal H\subseteq\mathcal G.$$

This property is one of the main reasons conditional expectation is central to martingale theory.

## 7. StochX objects

Use `FiniteProbabilitySpace`, `RandomVariable` and `Partition` together. The objects are deliberately explicit so probability and outcome validation happen before numerical computations.

See the [probability objects guide](stochastic/probability-objects.md) and corresponding API pages.
