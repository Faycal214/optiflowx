# Chapitre 3 — Chaînes de Markov à Temps Continu (CMTC)

> **Cours de référence :** N. Boussaha, *Processus Aléatoires (3)*, MSPRO, USTHB, 2024–2025.
>
> Cette page appartient à **Course material**. Elle suit le support du chapitre et reste séparée de la documentation Package / API.

## 1. Définition d'une CMTC

Sur un espace de probabilité, un processus \(X=(X_t)_{t\ge0}\) à espace d'états fini ou dénombrable est une chaîne de Markov à temps continu lorsque, pour des instants

\[
0=t_0<t_1<\cdots<t_n<t_{n+1},
\]

il vérifie

\[
P(X_{t_{n+1}}=j_{n+1}\mid X_{t_n}=j_n,\ldots,X_{t_0}=j_0)
=
P(X_{t_{n+1}}=j_{n+1}\mid X_{t_n}=j_n).
\]

Dans le cas homogène,

\[
P(X_{s+t}=j\mid X_s=i)=p_{ij}(t),
\]

donc la transition dépend uniquement de la durée \(t\). Le support note

\[
P(t)=(p_{ij}(t)),
\qquad p_{ij}(0)=\delta_{ij},
\]

avec \(p_{ij}(t)\ge0\) et des lignes de somme \(1\). fileciteturn478file3L219-L250

## 2. Générateur infinitésimal

Le générateur \(Q=(q_{ij})\) décrit le comportement pendant un intervalle infinitésimal \(h\). Pour \(i\neq j\),

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

La probabilité de deux transitions ou plus sur un intervalle infinitésimal est \(o(h)\). fileciteturn468file2L1633-L1649 fileciteturn468file2L1651-L1692

## 3. Processus de Poisson comme CMTC

Le support reprend le processus de Poisson de taux \(\lambda\) et obtient un générateur dont les seules transitions possibles sont \(i\to i+1\), avec taux \(\lambda\), et une diagonale compensatrice. fileciteturn468file2L1861-L1917

## 4. Équations de Kolmogorov

Les équations de Kolmogorov relient l'évolution de \(P(t)\) au générateur. Pour un espace d'états fini, le support donne notamment l'équation backward

\[
P'(t)=QP(t),
\qquad P(0)=I,
\]

et sa solution matricielle

\[
\boxed{P(t)=e^{tQ}}.
\]

L'exponentielle matricielle est définie par

\[
e^{tQ}=I+tQ+\frac{t^2Q^2}{2!}+\cdots
=\sum_{k=0}^{\infty}\frac{t^kQ^k}{k!}.
\]

Le support souligne cependant que, dans le cadre général, notamment lorsque \(S\) est infini, une expression explicite de \(P(t)\) n'est pas toujours disponible. fileciteturn478file2L149-L195

## 5. Loi de l'état

Si \(\mu_0\) est la loi initiale sous forme de vecteur ligne, alors

\[
\mu_t=\mu_0P(t).
\]

Cette identité est l'analogue continu du résultat discret \(\mu_n=\mu_0P^n\).

## 6. Distribution stationnaire

Une distribution \(\pi\) est stationnaire si

\[
\pi P(t)=\pi,
\qquad \forall t\ge0.
\]

Le support établit l'équivalence, dans le cadre considéré,

\[
\boxed{\pi P(t)=\pi\ \forall t\ge0
\iff
\pi Q=0,\quad \sum_i\pi_i=1,\quad \pi_i\ge0.}
\]

L'intérêt est pratique : lorsque \(P(t)\) ne peut pas être calculée explicitement, on peut chercher \(\pi\) directement à partir de \(Q\). fileciteturn478file6L424-L440

## 7. Temps de séjour

Le processus reste un temps aléatoire dans son état courant avant d'effectuer un saut vers un nouvel état. Le support décrit cette construction par des temps de séjour successifs : à partir de \(X_0\), le processus attend un premier temps aléatoire, saute, attend à nouveau, et ainsi de suite. fileciteturn478file0L45-L58

Le taux total de sortie de \(i\) est

\[
-q_{ii}=\sum_{j\ne i}q_{ij}.
\]

## 8. Chaîne embarquée

Les états observés aux instants de saut forment une chaîne de Markov discrète, appelée chaîne embarquée. Lorsque \(-q_{ii}>0\), la probabilité que le prochain saut depuis \(i\) arrive en \(j\neq i\) est obtenue à partir des taux de sortie par

\[
r_{ij}=\frac{q_{ij}}{-q_{ii}}.
\]

Cette construction permet de réutiliser les résultats des CMTD pour analyser certains aspects de la CMTC.

## 9. Comportement de long terme

Le chapitre étudie ensuite les propriétés de retour, les proportions d'occupation et le comportement asymptotique. Pour un coût \(h(i)\) par unité de temps dans l'état \(i\), le coût moyen selon une distribution stationnaire \(\pi\) est

\[
\sum_i\pi_i h(i).
\]

fileciteturn469file3L98-L106

Le support donne également des résultats sur la non-explosion et la distribution limite : pour une CMTC irréductible non explosive et récurrente positive, il existe une distribution stationnaire unique \(\pi\) vérifiant \(\pi Q=0\) et

\[
\lim_{t\to\infty}p_{ij}(t)=\pi_j.
\]

Il relie aussi le temps moyen de retour à la distribution stationnaire par

\[
\mu_i=\frac{1}{q_{ii}\pi_i}
\]

avec la convention du cours sur le signe de \(q_{ii}\) et son taux de sortie. fileciteturn478file7L466-L495

Le support donne enfin un critère de non-explosion pour un générateur associé à une CMTC : un espace d'états fini et un ensemble de taux diagonaux borné conduisent au cadre non explosif étudié dans le théorème correspondant. fileciteturn478file7L461-L466

# 10. Processus de naissance et de mort

Un processus de naissance et de mort est un cas particulier de CMTC où les seuls sauts possibles sont

\[
i\to i+1 \quad\text{(naissance)},
\]

et, pour \(i>0\),

\[
i\to i-1 \quad\text{(mort)}.
\]

fileciteturn469file3L104-L113

On introduit les taux de naissance \(\lambda_i\) et de mort \(\mu_i\). Le générateur est tridiagonal et le diagramme de transition porte ces taux, pas des probabilités finies. fileciteturn469file3L115-L145

## 10.1. Équations de Kolmogorov

En posant

\[
p_k(t)=P(X_t=k),
\]

les équations de Kolmogorov expriment le bilan des flux entrants et sortants de chaque état \(k\). Le support traite ces équations explicitement pour les processus de naissance et de mort. fileciteturn469file3L147-L158

## 10.2. Distribution stationnaire

Lorsque la distribution stationnaire existe et est normalisable, les masses sont reliées aux taux successifs. La forme récursive conduit à une construction de type

\[
\pi_n\propto\prod_{k=0}^{n-1}\frac{\lambda_k}{\mu_{k+1}},
\]

puis à une normalisation par \(\sum_n\pi_n=1\).

## 10.3. Taux linéaires

Le support étudie notamment

\[
\lambda_n=n\lambda+\alpha,
\]

où \(\alpha\ge0\) est un taux d'immigration constant et \(\lambda\ge0\) le taux de naissance par individu. Avec \(n\) individus, une mort sur \([t,t+h[\) a une probabilité de l'ordre de \(n\mu h\). fileciteturn469file3L164-L185

## 10.4. Non-explosion

Le cours étudie le temps d'explosion \(\zeta\). Pour une naissance pure, il donne notamment le critère

\[
\sum_k\frac1{\lambda_k}=\infty
\quad\Longrightarrow\quad
P(\zeta=\infty)=1.
\]

fileciteturn469file3L158-L162

## 11. Synthèse

La progression du chapitre est donc

\[
\text{définition CMTC}
\rightarrow P(t)
\rightarrow Q
\rightarrow\text{Kolmogorov}
\rightarrow\mu_t
\rightarrow\text{stationnarité}
\rightarrow\text{séjours}
\rightarrow\text{chaîne embarquée}
\rightarrow\text{long terme},
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

Cette page reste limitée au contenu du chapitre 3 fourni.