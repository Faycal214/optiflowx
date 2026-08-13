# Chapitre 3 — Chaînes de Markov à Temps Continu (CMTC)

> **Cours de référence :** N. Boussaha, *Processus Aléatoires (3)*, MSPRO, USTHB, 2024–2025.
>
> Cette page appartient à **Course material**. Elle suit le support du chapitre et reste séparée de la documentation Package / API.

## 1. CMTC et homogénéité

Une chaîne de Markov à temps continu est un processus de Markov indexé par un temps continu. Dans le cas homogène, on écrit

\[
p_{ij}(t)=P(X_{s+t}=j\mid X_s=i),
\qquad P(t)=(p_{ij}(t))_{i,j\in S}.
\]

On a \(P(0)=I\), et les probabilités de transition décrivent l'évolution du processus pendant une durée donnée.

## 2. Générateur infinitésimal

Le chapitre introduit la matrice \(Q=(q_{ij})\). Pour \(i\neq j\),

\[
q_{ij}=\lim_{h\to0}\frac{p_{ij}(h)}{h},
\]

et

\[
q_{ii}=-\sum_{j\neq i}q_{ij}.
\]

Ainsi,

\[
p_{ij}(h)=q_{ij}h+o(h),\quad i\neq j,
\]

et

\[
p_{ii}(h)=1+q_{ii}h+o(h).
\]

Chaque ligne vérifie \(\sum_jq_{ij}=0\). Le support précise également que la probabilité de deux transitions ou plus dans un intervalle infinitésimal est \(o(h)\). fileciteturn468file2L1633-L1649 fileciteturn468file2L1651-L1692

Les arcs du diagramme de transition portent des **taux** et non des probabilités de transition sur une durée finie. fileciteturn469file3L139-L145

## 3. Exemple du processus de Poisson

Le processus de Poisson de taux \(\lambda\) est repris comme exemple de CMTC. Le calcul infinitésimal donne un taux de transition \(\lambda\) de \(i\) vers \(i+1\), avec la diagonale compensatrice du générateur. fileciteturn468file2L1861-L1917

## 4. Équations de Kolmogorov

Le générateur gouverne l'évolution de \(P(t)\) par les équations de Kolmogorov avant et arrière. Dans le cadre matriciel développé dans le cours,

\[
P(t)=e^{tQ}.
\]

Le support remarque toutefois qu'une forme explicite n'est pas toujours disponible dans le cas général, notamment lorsque l'espace des états est infini. fileciteturn469file2L13-L16

## 5. Loi de l'état

Si \(\mu_0\) est la loi initiale sous forme de vecteur ligne, alors

\[
\mu_t=\mu_0P(t).
\]

C'est l'analogue continu de \(\mu_n=\mu_0P^n\) en temps discret.

## 6. Distribution stationnaire

Une distribution \(\pi\) est stationnaire si

\[
\pi P(t)=\pi,\qquad \forall t\ge0.
\]

Le cours montre que l'on peut alors rechercher \(\pi\) en résolvant

\[
\boxed{\pi Q=0},
\]

avec

\[
\pi_i\ge0,
\qquad \sum_i\pi_i=1.
\]

Cette formulation est particulièrement utile lorsque \(P(t)\) ne peut pas être calculée explicitement. fileciteturn469file2L18-L30

## 7. Temps de séjour, chaîne embarquée et long terme

Le taux total de sortie de l'état \(i\) est \(-q_{ii}\). Les temps de séjour dans les états sont étudiés à partir de ce taux. Les états visités au moment des sauts forment la **chaîne embarquée**, qui permet de relier les propriétés continues aux propriétés d'une chaîne discrète.

Le chapitre étudie ensuite les temps de retour, les proportions de temps d'occupation et les résultats de long terme. Une application introduite dans le support attribue un coût par unité de temps \(h(i)\) à chaque état ; sous une loi stationnaire \(\pi\), le coût moyen est la moyenne pondérée

\[
\sum_i\pi_i h(i).
\]

fileciteturn469file3L98-L106

# 8. Processus de naissance et de mort

Un processus de naissance et de mort est un cas particulier de CMTC où les seuls sauts sont

\[
i\to i+1 \quad\text{(naissance)},
\]

et, pour \(i>0\),

\[
i\to i-1 \quad\text{(mort)}.
\]

fileciteturn469file3L104-L113

On introduit les taux de naissance \(\lambda_i\) et de mort \(\mu_i\). Le générateur est tridiagonal et le diagramme porte ces taux. fileciteturn469file3L115-L145

## 8.1. Équations de Kolmogorov

En posant \(p_k(t)=P(X_t=k)\), le chapitre écrit les équations d'évolution sous forme de bilan des flux entrants et sortants de l'état \(k\). fileciteturn469file3L147-L158

## 8.2. Stationnarité

À l'équilibre, les masses stationnaires sont reliées aux taux successifs de naissance et de mort. Lorsque la normalisation est possible, on obtient une relation produit de la forme

\[
\pi_n\propto\prod_{k=0}^{n-1}\frac{\lambda_k}{\mu_{k+1}},
\]

puis la normalisation \(\sum_n\pi_n=1\).

## 8.3. Taux linéaires

Le support traite notamment

\[
\lambda_n=n\lambda+\alpha,
\]

avec \(\alpha\ge0\) comme taux d'immigration constant et \(\lambda\ge0\) comme taux de naissance par individu. Le taux de mort est proportionnel à la population, de sorte qu'avec \(n\) individus la probabilité d'une mort pendant \([t,t+h[\) est de l'ordre de \(n\mu h\). fileciteturn469file3L164-L185

## 8.4. Non-explosion

Le cours étudie enfin le temps d'explosion, c'est-à-dire la possibilité d'une infinité de sauts en temps fini. Pour le processus de naissance pure, il donne notamment le critère

\[
\sum_k\frac1{\lambda_k}=\infty
\quad\Longrightarrow\quad
P(\zeta=\infty)=1.
\]

fileciteturn469file3L158-L162

## 9. Synthèse

Le chapitre suit donc la progression

\[
\text{CMTC}\rightarrow P(t)\rightarrow Q\rightarrow\text{Kolmogorov}\rightarrow\mu_t\rightarrow\text{stationnarité}\rightarrow\text{séjours / chaîne embarquée}\rightarrow\text{long terme},
\]

puis développe le cas particulier

\[
\text{naissance--mort}\rightarrow\text{générateur}\rightarrow\text{Kolmogorov}\rightarrow\text{stationnarité}\rightarrow\text{cas particuliers et explosion}.
\]

Cette page reste limitée au contenu du chapitre 3 fourni.