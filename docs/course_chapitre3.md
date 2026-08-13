# Chapitre 3 — Chaînes de Markov à Temps Continu (CMTC)

> **Cours de référence :** N. Boussaha, *Processus Aléatoires (3)*, Master Modélisation Stochastique et Prévision en Recherche Opérationnelle (MSPRO), USTHB, 2024–2025.
>
> Cette page appartient à **Course material**. Elle suit le support du chapitre et reste séparée de la documentation Package / API.

## 1. Définition d'une CMTC

Sur un espace de probabilité, un processus \(X=(X_t)_{t\ge0}\) à espace d'états fini ou dénombrable est une chaîne de Markov à temps continu lorsque, pour des instants

\[
0=t_0<t_1<\cdots<t_n<t_{n+1},
\]

\[
P(X_{t_{n+1}}=j_{n+1}\mid X_{t_n}=j_n,\ldots,X_{t_0}=j_0)
=P(X_{t_{n+1}}=j_{n+1}\mid X_{t_n}=j_n).
\]

Dans le cas homogène,

\[
P(X_{s+t}=j\mid X_s=i)=p_{ij}(t),
\]

la transition dépendant uniquement de la durée \(t\). On note

\[
P(t)=(p_{ij}(t)),
\qquad p_{ij}(0)=\delta_{ij}.
\]

Chaque ligne de \(P(t)\) est une distribution de probabilité.

## 2. Générateur infinitésimal

Le générateur \(Q=(q_{ij})\) décrit le comportement pendant une durée infinitésimale \(h\). Pour \(i\ne j\),

\[
q_{ij}=\lim_{h\to0}\frac{p_{ij}(h)}{h},
\]

et

\[
p_{ij}(h)=q_{ij}h+o(h),
\qquad
p_{ii}(h)=1+q_{ii}h+o(h).
\]

Les lignes de \(Q\) ont somme nulle :

\[
\sum_jq_{ij}=0,
\qquad
q_{ii}=-\sum_{j\ne i}q_{ij}.
\]

La probabilité d'effectuer deux transitions ou plus sur un intervalle infinitésimal est d'ordre \(o(h)\).

## 3. Processus de Poisson comme CMTC

Pour un processus de Poisson de taux \(\lambda\), les transitions possibles sont \(i\to i+1\) avec taux \(\lambda\), et la diagonale de \(Q\) compense les taux de sortie. Le processus de Poisson fournit ainsi un exemple élémentaire de CMTC.

## 4. Équations de Kolmogorov

Le générateur gouverne l'évolution de la matrice de transition. Dans le cas matriciel fini, l'équation backward s'écrit

\[
P'(t)=QP(t),
\qquad P(0)=I,
\]

et donne

\[
\boxed{P(t)=e^{tQ}}.
\]

L'exponentielle matricielle est définie par

\[
e^{tQ}=I+tQ+\frac{t^2Q^2}{2!}+\cdots
=\sum_{k=0}^{\infty}\frac{t^kQ^k}{k!}.
\]

Dans le cas général, surtout lorsque l'espace des états est infini, une expression explicite de \(P(t)\) peut ne pas être disponible.

## 5. Loi de l'état

Si \(\mu_0\) est la loi initiale sous forme de vecteur ligne,

\[
\boxed{\mu_t=\mu_0P(t)}.
\]

C'est l'analogue continu de \(\mu_n=\mu_0P^n\) pour les chaînes à temps discret.

## 6. Distribution stationnaire

Une distribution \(\pi\) est stationnaire si

\[
\pi P(t)=\pi,
\qquad \forall t\ge0.
\]

Le support établit la caractérisation par le générateur :

\[
\boxed{\pi Q=0},
\qquad
\pi_i\ge0,
\qquad
\sum_i\pi_i=1.
\]

Cette équation permet de trouver une distribution stationnaire sans calculer explicitement toute la matrice \(P(t)\).

## 7. Temps de séjour

Le processus reste pendant un temps aléatoire dans son état courant avant de sauter vers un nouvel état. Les temps de séjour successifs décrivent la trajectoire entre les instants de saut.

Le taux total de sortie de l'état \(i\) est

\[
-q_{ii}=\sum_{j\ne i}q_{ij}.
\]

## 8. Chaîne embarquée

Les états observés aux instants de saut constituent une chaîne de Markov à temps discret, appelée chaîne embarquée. Lorsque \(-q_{ii}>0\), la probabilité que le prochain saut depuis \(i\) conduise à \(j\ne i\) est

\[
r_{ij}=\frac{q_{ij}}{-q_{ii}}.
\]

Cette construction relie les propriétés de la CMTC à celles des CMTD.

## 9. Retour, occupation et comportement asymptotique

Le chapitre étudie les temps de retour et les proportions de temps passées dans les états. Si chaque unité de temps passée dans l'état \(i\) engendre un coût \(h(i)\), alors sous une distribution stationnaire \(\pi\), le coût moyen est

\[
\sum_i\pi_i h(i).
\]

Pour une CMTC irréductible non explosive et récurrente positive, le support donne une distribution stationnaire unique \(\pi\) telle que

\[
\pi Q=0,
\]

et

\[
\lim_{t\to\infty}p_{ij}(t)=\pi_j.
\]

Il relie également le temps moyen de retour à la distribution stationnaire par une formule faisant intervenir le taux de sortie \(-q_{ii}\) et \(\pi_i\).

Le cours traite aussi la non-explosion, c'est-à-dire l'absence d'une infinité de sauts en temps fini.

# 10. Processus de naissance et de mort

Un processus de naissance et de mort est un cas particulier de CMTC où les seuls sauts possibles sont

\[
i\to i+1 \quad\text{(naissance)},
\]

et, pour \(i>0\),

\[
i\to i-1 \quad\text{(mort)}.
\]

On introduit les taux de naissance \(\lambda_i\) et de mort \(\mu_i\). Le générateur est alors tridiagonal et le diagramme de transition porte ces taux.

## 10.1. Équations de Kolmogorov

En posant

\[
p_k(t)=P(X_t=k),
\]

les équations de Kolmogorov décrivent le bilan des flux entrants et sortants de chaque état \(k\).

## 10.2. Distribution stationnaire

Lorsque la distribution stationnaire existe, les probabilités stationnaires sont liées aux taux successifs par une relation récursive de la forme

\[
\pi_n\propto\prod_{k=0}^{n-1}\frac{\lambda_k}{\mu_{k+1}},
\]

suivie d'une normalisation lorsque la somme des masses est finie.

## 10.3. Taux linéaires

Le support étudie notamment

\[
\lambda_n=n\lambda+\alpha,
\]

où \(\alpha\ge0\) représente l'immigration et \(\lambda\ge0\) le taux de naissance par individu. Le taux de mort est proportionnel au nombre d'individus : avec \(n\) individus, une mort pendant \([t,t+h[\) est de probabilité de l'ordre de \(n\mu h\).

## 10.4. Non-explosion

Le cours étudie le temps d'explosion \(\zeta\). Pour une naissance pure, un critère présenté est

\[
\sum_k\frac1{\lambda_k}=\infty
\quad\Longrightarrow\quad
P(\zeta=\infty)=1.
\]

## 11. Synthèse

\[
\text{CMTC}
\rightarrow P(t)
\rightarrow Q
\rightarrow\text{Kolmogorov}
\rightarrow\mu_t
\rightarrow\text{stationnarité}
\rightarrow\text{séjours}
\rightarrow\text{chaîne embarquée}
\rightarrow\text{long terme}
\]

puis

\[
\text{naissance--mort}
\rightarrow\text{générateur}
\rightarrow\text{Kolmogorov}
\rightarrow\text{stationnarité}
\rightarrow\text{taux particuliers}
\rightarrow\text{non-explosion}.
\]