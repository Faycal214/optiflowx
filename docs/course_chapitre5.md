# Chapitre 5 — Généralités sur les martingales à temps discret

> **Source de référence :** N. Boussaha, *Processus Aléatoires (5)*, Master Modélisation Stochastique et Prévision en Recherche Opérationnelle (MSPRO), USTHB, 2024–2025.
>
> Cette page appartient à **Course material**. Elle suit la progression mathématique du chapitre fourni et reste séparée de la documentation Package / API et des exemples.

## 1. Introduction

Le chapitre introduit les martingales dans le cadre du temps discret. Le support les présente à partir de l'idée de jeu équitable et place ensuite les sous-martingales et surmartingales du côté des jeux favorables ou défavorables. Le chapitre insiste sur les applications en probabilités modernes et en analyse stochastique. fileciteturn423file0L17-L30

## 2. Filtrations et processus adaptés

### Filtration
Une filtration de $(\Omega,\mathcal A,P)$ est une suite croissante de sous-tribus $(\mathcal F_n)_{n\in\mathbb N}$ de $\mathcal A$ :

$$\mathcal F_0\subseteq\mathcal F_1\subseteq\cdots\subseteq\mathcal A.$$

Le quadruplet $(\Omega,\mathcal A,(\mathcal F_n),P)$ est alors un espace probabilisé filtré. fileciteturn423file0L100-L108

### Processus adapté
Un processus $(X_n)$ est adapté à $(\mathcal F_n)$ lorsque, pour tout $n$, $X_n$ est mesurable par rapport à $\mathcal F_n$. fileciteturn423file0L105-L108

### Filtration naturelle
Pour un processus $(X_n)$, la filtration naturelle est

$$\mathcal F_n^X=\sigma(X_0,\ldots,X_n).$$

Le cours la décrit comme la plus petite filtration qui rende le processus adapté. Si $(X_n)$ est adapté à une filtration $(\mathcal F_n)$, alors $X_n$ reste mesurable par rapport à $\mathcal F_m$ pour tout $m\ge n$. fileciteturn423file0L119-L136

## 3. Martingale

Soit $(X_n)_{n\ge0}$ un processus défini sur $(\Omega,\mathcal A,(\mathcal F_n),P)$. Le cours définit une martingale relativement à $(\mathcal F_n)$ par les trois conditions, pour tout $n$ :

1. $X_n$ est $\mathcal F_n$-mesurable ;
2. $E[X_{n+1}\mid\mathcal F_n]=X_n$ presque sûrement ;
3. $E(|X_n|)<+\infty$.

fileciteturn423file0L158-L167

La condition centrale peut aussi s'écrire

$$E[(X_{n+1}-X_n)\mid\mathcal F_n]=0,$$

ou, pour tout $A\in\mathcal F_n$,

$$E[1_A X_{n+1}]=E[1_A X_n].$$

Lorsque la filtration n'est pas précisée, le support considère la filtration naturelle. fileciteturn423file0L169-L180

## 4. Martingale de Doob

Si $(\mathcal F_n)$ est une filtration et $X$ une variable aléatoire intégrable, le processus

$$X_n=E[X\mid\mathcal F_n]$$

est une martingale : c'est la **martingale de Doob**. Le cours vérifie son intégrabilité, son adaptation et la propriété de martingale par emboîtement de l'espérance conditionnelle. fileciteturn424file0L11-L32

## 5. Inégalité de Jensen et constructions

Pour une fonction convexe $\varphi$,

$$\varphi(E[X])\le E[\varphi(X)],$$

et l'inégalité est inversée pour une fonction concave. fileciteturn425file7L467-L472

Pour un processus intégrable et adapté $(\varepsilon_n)$, le cours considère

$$X_n=\varepsilon_1+\cdots+\varepsilon_n.$$

Cette somme est une martingale si et seulement si

$$E[\varepsilon_{n+1}\mid\mathcal F_n]=0.$$

Des incréments indépendants et centrés donnent en particulier une martingale relativement à la filtration naturelle. fileciteturn424file0L42-L50 fileciteturn425file7L473-L486

Le chapitre propose aussi des exercices autour de la marche symétrique $X_n=\sum_{i=1}^n\xi_i$, avec $\xi_i$ indépendantes et uniformes sur $\{-1,1\}$, ainsi que des transformations exponentielles. fileciteturn425file8L507-L545

## 6. Propriétés élémentaires

Si $(X_n)$ est une martingale, alors

$$E[X_n]=E[X_0].$$

L'espérance est donc constante. fileciteturn408file0L378-L395

Le support donne également

$$E[X_{n+k}\mid\mathcal F_n]=X_n,$$

ou, pour $m<n$,

$$E[X_n\mid\mathcal F_m]=X_m.$$ 

fileciteturn408file0L398-L410

## 7. Surmartingales et sous-martingales

Un processus $(X_n)$ est une **surmartingale** s'il est intégrable, adapté et vérifie

$$E[X_{n+1}\mid\mathcal F_n]\le X_n\quad\text{p.s.}$$

Une **sous-martingale** vérifie l'inégalité opposée. Une martingale est simultanément une surmartingale et une sous-martingale. fileciteturn424file1L127-L135

Le cours considère une marche biaisée : avec probabilité $p$, un incrément vaut $+1$ et avec probabilité $1-p$, il vaut $-1$. Elle est une martingale pour $p=1/2$, une surmartingale pour $p<1/2$ et une sous-martingale pour $p>1/2$. fileciteturn424file2L146-L158

## 8. Transformations et monotonie des espérances

Si $(X_n)$ est une martingale et $\varphi$ une fonction convexe donnant des variables intégrables, le processus $(\varphi(X_n))$ est une sous-martingale. Les versions correspondant aux fonctions convexes croissantes et aux fonctions concaves sont également énoncées dans le support. fileciteturn424file2L160-L182

Le cours en déduit notamment que $|X_n|$ et, lorsque l'intégrabilité est assurée, $X_n^2$ sont des sous-martingales pour une martingale. La partie positive d'une sous-martingale est elle-même une sous-martingale. fileciteturn424file2L184-L184

Les espérances d'une surmartingale sont décroissantes et celles d'une sous-martingale sont croissantes. Les inégalités à plusieurs pas sont également données. fileciteturn424file4L867-L873

## 9. Temps d'arrêt

### Motivation

Le cours introduit un temps d'arrêt par une décision d'arrêter un jeu ou un processus à partir des informations disponibles à chaque étape. À l'instant $n$, on doit être capable de décider si l'arrêt a lieu à cet instant. fileciteturn426file5L358-L380

### Définition

Dans l'espace probabilisé filtré, une variable aléatoire

$$T:\Omega\to\mathbb N\cup\{+\infty\}$$

est un **temps d'arrêt** si, pour tout $n\in\mathbb N$,

$$\{T=n\}\in\mathcal F_n.$$

Le cours indique que cette condition est équivalente à

$$\{T\le n\}\in\mathcal F_n,$$

ou encore à

$$\{T>n\}\in\mathcal F_n.$$

La valeur $+\infty$ est autorisée. fileciteturn426file5L382-L397 fileciteturn426file6L421-L433

### Stabilité

Si $S$ et $T$ sont des temps d'arrêt relativement à la même filtration, alors

$$S+T,\qquad S\wedge T,\qquad S\vee T$$

sont aussi des temps d'arrêt. Pour une suite de temps d'arrêt, le cours donne également la stabilité de $\inf T_k$, $\sup T_k$, $\liminf T_k$ et $\limsup T_k$. fileciteturn426file3L241-L250

### Premier temps d'atteinte

Dans l'exemple de la fortune d'un joueur, l'arrêt est donné par le premier instant où la fortune atteint l'une de deux barrières. Cela fournit un exemple de premier temps d'atteinte et de temps d'arrêt. fileciteturn426file7L506-L515

## 10. Processus arrêté

Soit $(X_n)$ un processus adapté à $(\mathcal F_n)$ et $\tau$ un temps d'arrêt pour cette filtration. Le **processus arrêté** est noté $(X_n^\tau)_{n\ge0}$ et défini par

$$
X_n^\tau(\omega)=
\begin{cases}
X_n(\omega),&n<\tau(\omega),\\
X_{\tau(\omega)}(\omega),&n\ge\tau(\omega).
\end{cases}
$$

Une écriture équivalente est

$$
X_n^\tau=X_n1_{\{n<\tau\}}+X_\tau1_{\{\tau\le n\}}.
$$

Le support remarque que le processus arrêté reste adapté. fileciteturn425file3L151-L174

Il donne aussi une écriture à partir des incréments :

$$
X_n^\tau=X_0+\sum_{k=0}^{n-1}(X_{k+1}-X_k)1_{\{\tau>k\}}.
$$

fileciteturn425file3L176-L185

## 11. Une martingale arrêtée reste une martingale

Le résultat central est : si $(X_n)$ est une martingale et $\tau$ un temps d'arrêt relatif à $(\mathcal F_n)$, alors

$$
(X_n^\tau)_{n\ge0}
$$

est encore une martingale. fileciteturn425file1L58-L74

L'argument du cours sépare le cas où le temps d'arrêt est déjà atteint et le cas où le processus n'est pas encore arrêté. Après l'arrêt, l'incrément du processus arrêté est nul ; avant l'arrêt, on retrouve l'incrément de la martingale initiale. fileciteturn426file1L59-L73

## 12. Variable terminale du processus arrêté

Le support étudie ensuite la variable obtenue lorsque l'on remplace le temps déterministe $n$ par un temps d'arrêt $\tau$. Pour un temps d'arrêt borné, cette variable est la variable terminale du processus arrêté ; plus généralement, elle est considérée lorsque $\tau$ est presque sûrement fini. fileciteturn425file1L75-L86

Si

$$P(\tau<+\infty)=1,$$

le cours donne alors

$$
X_n^\tau\xrightarrow[n\to\infty]{\text{p.s.}}X^\tau.
$$

fileciteturn425file1L87-L98

## 13. Progression du chapitre

Le développement du support peut être lu comme la progression suivante :

$$
\text{Filtration}
\rightarrow
\text{processus adapté}
\rightarrow
\text{martingale}
\rightarrow
\text{propriétés d'espérance}
$$

puis

$$
\text{Jensen}
\rightarrow
\text{sous-martingale / surmartingale}
\rightarrow
\text{transformations}
$$

et enfin

$$
\text{temps d'arrêt}
\rightarrow
\text{processus arrêté}
\rightarrow
\text{martingale arrêtée}
\rightarrow
\text{variable terminale}.
$$

Cette page reste limitée aux notions et résultats développés dans le chapitre fourni. fileciteturn423file0L97-L108