# Chapter 4 — Conditional Expectation

This page develops conditional expectation from event conditioning through conditioning on random variables and sigma-fields. The mathematical discussion is separate from the Python API and worked examples.

## 1. Introduction

Conditional expectation is introduced as a tool for estimation when partial information is available, especially in prediction and when some data are unobserved or missing.

## 2. Conditioning on an event

For \(B\in\mathcal F\) such that \(P(B)>0\),

\[
P(A\mid B)=\frac{P(A\cap B)}{P(B)}.
\]

For \(X\in L^1\),

\[
\boxed{E(X\mid B)=\frac{E(X\mathbf 1_B)}{P(B)}}.
\]

### Example

Three coins with values 10, 20, and 50 are tossed. If \(X\) is the total amount obtained on heads and \(B\) is the event that exactly two coins show heads, then

\[
E(X\mid B)=\frac{160}{3}.
\]

## 3. Conditioning on a discrete random variable

Let \(Y\) take values in a countable space \(E\), and define

\[
E_0=\{y\in E:P(Y=y)>0\}.
\]

For \(y\in E_0\),

\[
E(X\mid Y=y)=\frac{E(X\mathbf1_{\{Y=y\}})}{P(Y=y)}.
\]

Then define

\[
E(X\mid Y)=g(Y),
\]

where \(g(y)=E(X\mid Y=y)\) on \(E_0\). The value of \(g\) outside \(E_0\) is irrelevant because that set has probability zero.

Thus \(E(X\mid Y)\) is a random variable and a function of \(Y\), so it is \(\sigma(Y)\)-measurable.

### Example with a die

For a fair die, let \(X(\omega)=\omega\), and let \(Y\) indicate whether the result is odd or even. Then

\[
E(X\mid Y)=3\mathbf1_{\{Y=1\}}+4\mathbf1_{\{Y=0\}}.
\]

## 4. Absolutely continuous case

When \(X\) and \(Y\) have a joint density,

\[
f_{Y\mid X=x}(y)=\frac{f_{X,Y}(x,y)}{f_X(x)}.
\]

Then

\[
E(Y\mid X=x)=\int y f_{Y\mid X=x}(y)\,dy.
\]

In the corresponding example, this procedure gives

\[
E(Y\mid X)=X\quad\text{a.s.}
\]

## 5. Properties in the discrete case

The main properties include

\[
E|E(X\mid Y)|\le E|X|,
\]

the law of total expectation

\[
E(X)=\sum_yE(X\mid Y=y)P(Y=y),
\]

and, under independence,

\[
E(X\mid Y=y)=E(X).
\]

For a function \(h\), the value of \(Y\) may be replaced by the corresponding constant inside the conditioning relation when appropriate.

## 6. Characterization through \(\sigma(Y)\)

For \(X\in L^1\) and discrete \(Y\), \(E(X\mid Y)\) is, up to a null set, the unique \(\sigma(Y)\)-measurable random variable satisfying

\[
\int_A E(X\mid Y)\,dP
=
\int_A X\,dP,
\qquad A\in\sigma(Y).
\]

This formulation leads to conditioning with respect to an arbitrary random variable and then with respect to a sigma-field.

## 7. Arbitrary random variable

For \(X\in L^1\) and an arbitrary random variable \(Y\), conditional expectation is defined by the same property: it is \(\sigma(Y)\)-measurable and

\[
\int_A E(X\mid Y)\,dP
=
\int_A X\,dP,
\qquad A\in\sigma(Y).
\]

The relevant object is the information sigma-field. In particular,

\[
\sigma(Y)=\sigma(Y')
\Longrightarrow
E(X\mid Y)=E(X\mid Y')\quad\text{a.s.}
\]

## 8. Conditioning with respect to a sigma-field

Let \(\mathcal G\subseteq\mathcal F\) be a sub-sigma-field. For \(X\in L^1\), the conditional expectation \(E(X\mid\mathcal G)\) is a \(\mathcal G\)-measurable random variable such that

\[
\int_AE(X\mid\mathcal G)\,dP
=
\int_AX\,dP,
\qquad A\in\mathcal G.
\]

When \(\mathcal G=\sigma(Y)\), this gives the notation \(E(X\mid Y)\).

## 9. Characterization theorem

There exists a unique variable \(Y\in L^1(\Omega,\mathcal G,P)\) such that, for every bounded \(\mathcal G\)-measurable variable \(Z\),

\[
E(ZX)=E(ZY).
\]

This variable is

\[
Y=E(X\mid\mathcal G).
\]

In particular,

\[
E(\mathbf1_AX)=E\big(\mathbf1_AE(X\mid\mathcal G)\big),
\qquad A\in\mathcal G.
\]

The existence proof is beyond the scope of this finite framework.

## 10. Fundamental properties

If \(X\) is \(\mathcal G\)-measurable,

\[
E(X\mid\mathcal G)=X\quad\text{a.s.}
\]

Linearity:

\[
E(aX+bY\mid\mathcal G)
=aE(X\mid\mathcal G)+bE(Y\mid\mathcal G).
\]

Positivity:

\[
X\ge0\Rightarrow E(X\mid\mathcal G)\ge0.
\]

Total expectation:

\[
E(E(X\mid\mathcal G))=E(X).
\]

Absolute-value control:

\[
|E(X\mid\mathcal G)|\le E(|X|\mid\mathcal G),
\]

and therefore \(E|E(X\mid\mathcal G)|\le E|X|\).

Monotonicity:

\[
X\le X'\Rightarrow E(X\mid\mathcal G)\le E(X'\mid\mathcal G)\quad\text{a.s.}
\]

Independence: if \(X\) is independent of \(\mathcal G\),

\[
E(X\mid\mathcal G)=E(X)\quad\text{a.s.}
\]

## 11. Measurable factor

If \(Y\) is \(\mathcal G\)-measurable and the required integrability conditions hold,

\[
\boxed{E(YX\mid\mathcal G)=Y\,E(X\mid\mathcal G).}
\]

## 12. Successive conditioning

If \(\mathcal G_1\subseteq\mathcal G_2\), then

\[
\boxed{
E(E(X\mid\mathcal G_2)\mid\mathcal G_1)=E(X\mid\mathcal G_1).
}
\]

If \(B\in\mathcal G\), one also has

\[
E(E(X\mid\mathcal G)\mid B)=E(X\mid B).
\]

## 13. Independence of sigma-fields

Two sub-sigma-fields \(\mathcal G_1\) and \(\mathcal G_2\) are independent if and only if, for every integrable random variable measurable with respect to \(\mathcal G_2\),

\[
E(X\mid\mathcal G_1)=E(X).
\]

For independent random variables \(X\) and \(Y\), in particular,

\[
E(X\mid Y)=E(X).
\]

However, this last equality by itself is not sufficient to establish independence.

## 14. Summary

\[
\text{event}
\rightarrow\text{discrete variable}
\rightarrow\text{arbitrary variable}
\rightarrow\text{sigma-field}
\]

\[
\rightarrow\text{characterization}
\rightarrow\text{properties}
\rightarrow\text{independence}
\rightarrow\text{tower property}.
\]
