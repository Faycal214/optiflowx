# Chapitre 5 — Généralités sur les martingales à temps discret

> **Source de référence :** N. Boussaha, *Processus Aléatoires (5)*, Master Modélisation Stochastique et Prévision en Recherche Opérationnelle (MSPRO), USTHB, 2024–2025.
>
> Cette page appartient à **Course material**. Elle suit la progression mathématique du chapitre fourni et reste séparée de la documentation Package / API et des exemples.

## 1. Introduction

Le chapitre introduit les martingales dans le cadre du temps discret. Le support les présente à partir de l'idée de jeu équitable et place ensuite les sous-martingales et surmartingales du côté des jeux favorables ou défavorables. Le chapitre insiste sur les applications en probabilités modernes et en analyse stochastique. fileciteturn423file0L17-L30

## 2. Filtrations et processus adaptés

### Filtration

Une filtration de $(\Omega,\mathcal A,P)$ est une suite croissante de sous-tribus $(\mathcal F_n)_{n\in\mathbb N}$ de $\mathcal A$ :

$$
\mathcal F_0\subseteq\mathcal F_1\subseteq\cdots\subseteq\mathcal A.
$$

Le quadruplet $(\Omega,\mathcal A,(\mathcal F_n),P)$ est alors un espace probabilisé filtré. fileciteturn423file0L100-L108

### Processus adapté

Un processus $(X_n)$ est adapté à $(\mathcal F_n)$ lorsque, pour tout $n$, $X_n$ est mesurable par rapport à $\mathcal F_n$. fileciteturn423file0L105-L108

### Filtration naturelle

Pour un processus $(X_n)$, la filtration naturelle est

$$
\mathcal F_n^X=\sigma(X_0,\ldots,X_n).
$$

Le cours la décrit comme la plus petite filtration qui rende le processus adapté. Si $(X_n)$ est adapté à une filtration $(\mathcal F_n)$, alors $X_n$ reste mesurable par rapport à $\mathcal F_m$ pour tout $m\ge n$. fileciteturn423file0L119-L136

## 3. Martingale

Soit $(X_n)_{n\ge0}$ un processus défini sur $(\Omega,\mathcal A,(\mathcal F_n),P)$. Le cours définit une martingale relativement à $(\mathcal F_n)$ par les trois conditions, pour tout $n$ :

1. $X_n$ est $\mathcal F_n$-mesurable ;
2. $E[X_{n+1}\mid\mathcal F_n]=X_n$ presque sûrement ;
3. $E(|X_n|)<+\infty$.

fileciteturn423file0L158-L167

La condition centrale peut aussi s'écrire

$$
E[(X_{n+1}-X_n)\mid\mathcal F_n]=0,
$$

ou, pour tout $A\in\mathcal F_n$,

$$
E[1_A X_{n+1}]=E[1_A X_n].
$$

Le support rappelle que, lorsque la filtration n'est pas précisée, la filtration naturelle est sous-entendue. fileciteturn423file0L169-L180

## 4. Martingale de Doob

Si $(\mathcal F_n)$ est une filtration et $X$ une variable aléatoire intégrable, le processus

$$
X_n=E[X\mid\mathcal F_n]
$$

est une martingale : c'est la **martingale de Doob**. Le cours vérifie successivement son intégrabilité, son adaptation et la propriété de martingale à l'aide de la propriété d'emboîtement de l'espérance conditionnelle. fileciteturn424file0L11-L32

## 5. Inégalité de Jensen et constructions

Le support rappelle que, pour une fonction convexe $\varphi$,

$$
\varphi(E[X])\le E[\varphi(X)],
$$

et que l'inégalité est inversée pour une fonction concave. fileciteturn425file7L467-L472

Le chapitre utilise ensuite cette inégalité pour étudier des transformations de martingales et construire des sous-martingales ou des surmartingales.

### Sommes de variables centrées

Pour un processus intégrable et adapté $(\varepsilon_n)$, le cours considère

$$
X_n=\varepsilon_1+\cdots+\varepsilon_n.
$$

Cette somme est une martingale si et seulement si

$$
E[\varepsilon_{n+1}\mid\mathcal F_n]=0.
$$

En particulier, des incréments indépendants et centrés donnent une martingale relativement à la filtration naturelle correspondante. fileciteturn424file0L42-L50 fileciteturn425file7L473-L486

### Marche aléatoire symétrique

Le chapitre propose des exercices autour de

$$
X_n=\sum_{i=1}^n\xi_i,
$$

où les incréments $\xi_i$ sont indépendants et distribués uniformément sur $\{-1,1\}$. Les questions portent notamment sur des transformations de cette marche et sur des processus exponentiels de la forme $\exp(\beta X_n-\varphi n)$. fileciteturn425file8L507-L545

## 6. Propriétés élémentaires

### Espérance constante

Si $(X_n)$ est une martingale, alors

$$
E[X_n]=E[X_0],\qquad n\ge0.
$$

La martingale possède donc une espérance mathématique constante. fileciteturn408file0L378-L395

### Espérance conditionnelle à plusieurs pas

Le cours établit aussi

$$
E[X_{n+k}\mid\mathcal F_n]=X_n,
\qquad n,k\ge0,
$$

ou, de manière équivalente, pour $m<n$,

$$
E[X_n\mid\mathcal F_m]=X_m.
$$

Le résultat est obtenu par récurrence et par emboîtement de l'espérance conditionnelle. fileciteturn408file0L398-L410

## 7. Surmartingales et sous-martingales

Un processus $(X_n)$ est une **surmartingale** relativement à $(\mathcal F_n)$ lorsqu'il est intégrable, adapté et satisfait

$$
E[X_{n+1}\mid\mathcal F_n]\le X_n\quad\text{p.s.}
$$

Il est une **sous-martingale** lorsque

$$
E[X_{n+1}\mid\mathcal F_n]\ge X_n\quad\text{p.s.}
$$

Une martingale est à la fois une surmartingale et une sous-martingale. fileciteturn424file1L127-L135

### Marche biaisée

Le cours considère une marche dont chaque incrément vaut $+1$ avec probabilité $p$ et $-1$ avec probabilité $1-p$. Il donne la classification suivante :

- $p=1/2$ : martingale ;
- $p<1/2$ : surmartingale ;
- $p>1/2$ : sous-martingale.

L'exemple est interprété comme la fortune d'un joueur qui gagne ou perd une unité à chaque lancer. fileciteturn424file2L146-L158

## 8. Transformations par Jensen

Le support énonce la proposition suivante. Si $(X_n)$ est une martingale et $\varphi$ est convexe, avec l'intégrabilité nécessaire, alors $(\varphi(X_n))$ est une sous-martingale. Dans les cas sous- et sur-martingale, la monotonicité de $\varphi$ intervient selon l'énoncé du cours. Pour les fonctions concaves, les rôles sont inversés. fileciteturn424file2L160-L182

Le cours en déduit notamment que pour une martingale, $|X_n|$ et, lorsque l'intégrabilité est assurée, $X_n^2$ sont des sous-martingales. Pour une sous-martingale, $X_n^+=\max(X_n,0)$ est également une sous-martingale. fileciteturn424file2L184-L184

### Monotonie des espérances

Si $(X_n)$ est une surmartingale, alors $(E[X_n])$ est décroissante ; si $(X_n)$ est une sous-martingale, alors $(E[X_n])$ est croissante. Le support donne aussi les inégalités à plusieurs pas correspondantes. fileciteturn424file4L867-L873
