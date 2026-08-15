# Chapter 5 — Discrete-Time Martingales

This page develops discrete-time filtrations, martingales, stopping times, and stopped processes. The mathematical discussion remains separate from the Python API and worked examples.

## 1. Filtrations and adapted processes

A filtration is an increasing sequence of sigma-fields

$$\mathcal F_0\subseteq\mathcal F_1\subseteq\cdots\subseteq\mathcal A.$$

A process \((X_n)\) is adapted if \(X_n\) is \(\mathcal F_n\)-measurable for every \(n\). Its natural filtration is

$$\mathcal F_n^X=\sigma(X_0,\ldots,X_n).$$

## 2. Martingale

An integrable and adapted process \((X_n)\) is a martingale if

$$\boxed{E(X_{n+1}\mid\mathcal F_n)=X_n\quad\text{a.s.}}$$

Equivalently,

$$E(X_{n+1}-X_n\mid\mathcal F_n)=0.$$

For every \(A\in\mathcal F_n\),

$$E(\mathbf1_A X_{n+1})=E(\mathbf1_A X_n).$$

## 3. Doob martingale

For \(X\in L^1\), the process

$$M_n=E(X\mid\mathcal F_n)$$

is a martingale. This is the Doob martingale.

## 4. Jensen's inequality and constructions

For a convex function \(\varphi\),

$$\varphi(E(X))\le E(\varphi(X)).$$

A sum

$$X_n=\varepsilon_1+\cdots+\varepsilon_n$$

is a martingale when

$$E(\varepsilon_{n+1}\mid\mathcal F_n)=0.$$ 

A central example is the symmetric random walk \(X_n=\sum_{i=1}^n\xi_i\), with independent increments taking values in \(\{-1,1\}\), together with exponential transformations.

## 5. Properties

For a martingale,

$$E(X_n)=E(X_0),$$

and, for \(m<n\),

$$E(X_n\mid\mathcal F_m)=X_m.$$

## 6. Submartingales and supermartingales

A supermartingale satisfies

$$E(X_{n+1}\mid\mathcal F_n)\le X_n,$$

while a submartingale satisfies

$$E(X_{n+1}\mid\mathcal F_n)\ge X_n.$$

For a biased random walk with step \(+1\) and probability \(p\), the process is a martingale when \(p=1/2\), a supermartingale when \(p<1/2\), and a submartingale when \(p>1/2\).

If \(\varphi\) is convex and the required integrability conditions hold, \((\varphi(X_n))\) is a submartingale whenever \((X_n)\) is a martingale. In particular, \((|X_n|)\) and, when integrable, \((X_n^2)\), are submartingales.

## 7. Stopping times

A variable

$$T:\Omega\to\mathbb N\cup\{+\infty\}$$

is a stopping time if

$$\{T=n\}\in\mathcal F_n,$$

which is equivalent to the conditions \(\{T\le n\}\in\mathcal F_n\) and \(\{T>n\}\in\mathcal F_n\).

If \(S\) and \(T\) are stopping times for the same filtration, the operations \(S+T\), \(S\wedge T\), and \(S\vee T\) also give stopping times. The first hitting time of a set is a fundamental example.

## 8. Stopped process

For an adapted process \((X_n)\) and a stopping time \(\tau\), the stopped process follows the original path until the stopping time and then remains at its stopped value. One representation is

$$X_n^\tau=X_n\mathbf1_{\{n<\tau\}}+X_\tau\mathbf1_{\{\tau\le n\}}.$$

Another form is

$$X_n^\tau=X_0+\sum_{k=0}^{n-1}(X_{k+1}-X_k)\mathbf1_{\{\tau>k\}}.$$ 

## 9. Stopped martingale

If \((X_n)\) is a martingale and \(\tau\) is a stopping time, then the stopped process \((X_n^\tau)\) is again a martingale.

## 10. Terminal variable

When \(P(\tau<\infty)=1\), the terminal variable of the stopped process satisfies

$$X_n^\tau\xrightarrow[n\to\infty]{\mathrm{a.s.}}X^\tau.$$

## 11. Summary

$$\text{Filtration}\rightarrow\text{adapted process}\rightarrow\text{martingale}\rightarrow\text{Jensen}$$

$$\rightarrow\text{sub/supermartingale}\rightarrow\text{stopping time}\rightarrow\text{stopped process}\rightarrow\text{stopped martingale}.$$ 
