# Chapitre 5 — Généralités sur les martingales à temps discret

> **Cours de référence :** N. Boussaha, *Processus Aléatoires (5)*, Master Modélisation Stochastique et Prévision en Recherche Opérationnelle (MSPRO), USTHB, 2024–2025.
>
> Cette page appartient à **Course material**. Elle suit la progression mathématique du chapitre fourni et reste séparée de la documentation Package / API et des exemples.

## 1. Filtrations et processus adaptés

Une filtration est une suite croissante de sous-tribus

$$\mathcal F_0\subseteq\mathcal F_1\subseteq\cdots\subseteq\mathcal A.$$

Un processus \((X_n)\) est adapté si \(X_n\) est \(\mathcal F_n\)-mesurable pour tout \(n\). Sa filtration naturelle est

$$\mathcal F_n^X=\sigma(X_0,\ldots,X_n).$$

## 2. Martingale

Un processus intégrable et adapté \((X_n)\) est une martingale si

$$\boxed{E(X_{n+1}\mid\mathcal F_n)=X_n\quad\text{p.s.}}$$

Équivalemment,

$$E(X_{n+1}-X_n\mid\mathcal F_n)=0.$$

Pour tout \(A\in\mathcal F_n\),

$$E(\mathbf1_A X_{n+1})=E(\mathbf1_A X_n).$$

## 3. Martingale de Doob

Pour \(X\in L^1\), le processus

$$M_n=E(X\mid\mathcal F_n)$$

est une martingale. C'est la martingale de Doob.

## 4. Jensen et constructions

Pour une fonction convexe \(\varphi\),

$$\varphi(E(X))\le E(\varphi(X)).$$

Une somme

$$X_n=\varepsilon_1+\cdots+\varepsilon_n$$

est une martingale lorsque

$$E(\varepsilon_{n+1}\mid\mathcal F_n)=0.$$ 

Le support étudie notamment la marche symétrique \(X_n=\sum_{i=1}^n\xi_i\), avec des incréments indépendants de valeurs \(\{-1,1\}\), ainsi que des transformations exponentielles.

## 5. Propriétés

Pour une martingale,

$$E(X_n)=E(X_0),$$

et, pour \(m<n\),

$$E(X_n\mid\mathcal F_m)=X_m.$$

## 6. Sous-martingales et surmartingales

Une surmartingale vérifie

$$E(X_{n+1}\mid\mathcal F_n)\le X_n,$$

et une sous-martingale

$$E(X_{n+1}\mid\mathcal F_n)\ge X_n.$$

Pour une marche biaisée de pas \(+1\) avec probabilité \(p\), elle est une martingale si \(p=1/2\), une surmartingale si \(p<1/2\) et une sous-martingale si \(p>1/2\).

Si \(\varphi\) est convexe et les intégrabilités nécessaires sont satisfaites, \((\varphi(X_n))\) est une sous-martingale pour une martingale \((X_n)\). En particulier, \((|X_n|)\) et, lorsque c'est intégrable, \((X_n^2)\), sont des sous-martingales.

## 7. Temps d'arrêt

Une variable

$$T:\Omega\to\mathbb N\cup\{+\infty\}$$

est un temps d'arrêt si

$$\{T=n\}\in\mathcal F_n,$$

ce qui est équivalent aux conditions \(\{T\le n\}\in\mathcal F_n\) et \(\{T>n\}\in\mathcal F_n\).

Si \(S\) et \(T\) sont des temps d'arrêt pour la même filtration, les opérations \(S+T\), \(S\wedge T\) et \(S\vee T\) donnent également des temps d'arrêt. Le premier temps d'atteinte d'un ensemble est un exemple fondamental.

## 8. Processus arrêté

Pour un processus adapté \((X_n)\) et un temps d'arrêt \(\tau\), le processus arrêté conserve la trajectoire jusqu'à l'instant d'arrêt puis reste à sa valeur d'arrêt. Une écriture est

$$X_n^\tau=X_n\mathbf1_{\{n<\tau\}}+X_\tau\mathbf1_{\{\tau\le n\}}.$$

Une autre forme est

$$X_n^\tau=X_0+\sum_{k=0}^{n-1}(X_{k+1}-X_k)\mathbf1_{\{\tau>k\}}.$$ 

## 9. Martingale arrêtée

Si \((X_n)\) est une martingale et \(\tau\) un temps d'arrêt, alors le processus arrêté \((X_n^\tau)\) est encore une martingale.

## 10. Variable terminale

Lorsque \(P(\tau<\infty)=1\), le support considère la variable terminale du processus arrêté et obtient

$$X_n^\tau\xrightarrow[n\to\infty]{\mathrm{p.s.}}X^\tau.$$

## 11. Progression du chapitre

$$\text{Filtration}\rightarrow\text{processus adapté}\rightarrow\text{martingale}\rightarrow\text{Jensen}$$

$$\rightarrow\text{sous/surmartingale}\rightarrow\text{temps d'arrêt}\rightarrow\text{processus arrêté}\rightarrow\text{martingale arrêtée}.$$