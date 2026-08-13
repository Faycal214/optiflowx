# Chapitre 1 — Chaînes de Markov à Temps Discret (CMTD)

> **Cours de référence :** N. Boussaha, *Processus Aléatoires (1)*, Master Modélisation Stochastique et Prévision en Recherche Opérationnelle (MSPRO), USTHB, 2024–2025.
>
> Cette page appartient à **Course material**. Elle présente le contenu mathématique du chapitre dans l'ordre du support. Les classes Python et leur utilisation sont documentées séparément dans **Package / API** et les calculs appliqués sont montrés dans **Examples**.

## 1. Processus aléatoires

Un processus aléatoire est une famille de variables aléatoires définies sur un même espace de probabilité et indexées par un paramètre, généralement le temps. Le support introduit une application

\[
X:\Omega\times T\longrightarrow E,
\qquad (\omega,t)\longmapsto X_t(\omega),
\]

avec, pour chaque instant \(t\), une variable aléatoire \(X_t\). Le cours distingue les processus selon que le temps et l'espace des états sont discrets ou continus. La CMTD correspond au cas temps discret / espace des états discret.

## 2. Chaîne de Markov homogène

Soit \((X_n)_{n\in\mathbb N}\) un processus à temps discret et à espace d'états discret \(S\). La propriété de Markov est

\[
P(X_{n+1}=j\mid X_0=i_0,\ldots,X_n=i)=P(X_{n+1}=j\mid X_n=i).
\]

La chaîne est homogène lorsque les probabilités de transition ne dépendent pas de l'instant :

\[
P(X_{n+1}=j\mid X_n=i)=P(X_1=j\mid X_0=i).
\]

On note

\[
p_{ij}=P(X_{n+1}=j\mid X_n=i).
\]

## 3. Matrice de transition et graphe associé

Les probabilités sont regroupées dans

\[
P=(p_{ij})_{i,j\in S},
\]

avec

\[
p_{ij}\ge0,
\qquad \sum_{j\in S}p_{ij}=1.
\]

Le graphe associé est orienté : les états sont les sommets et une transition possible est représentée par une arête pondérée par sa probabilité.

### Marche aléatoire sur \(\mathbb Z\)

Le support considère notamment

\[
p_{ij}=\begin{cases}
p,&j=i+1,\\
q,&j=i-1,\\
r,&j=i,\\
0,&\text{sinon},
\end{cases}
\qquad p+q+r=1.
\]

## 4. Loi initiale et caractérisation

La loi initiale est

\[
\mu_0=(P(X_0=i))_{i\in S}.
\]

Une chaîne homogène est entièrement caractérisée par \(P\) et \(\mu_0\). Pour un chemin \(i_0,\ldots,i_n\),

\[
P(X_0=i_0,\ldots,X_n=i_n)
=\mu_0(i_0)p_{i_0i_1}\cdots p_{i_{n-1}i_n}.
\]

## 5. Transitions en plusieurs étapes

La probabilité de passage de \(i\) à \(j\) en \(n\) transitions est

\[
p_{ij}^{(n)}=P(X_{m+n}=j\mid X_m=i),
\]

et

\[
P^{(n)}=(p_{ij}^{(n)})_{i,j\in S}.
\]

La matrice \(P^{(n)}\) est stochastique.

## 6. Équations de Chapman–Kolmogorov

Pour \(m,n\ge0\),

\[
p_{ij}^{(n+m)}=\sum_{k\in S}p_{ik}^{(m)}p_{kj}^{(n)}.
\]

Sous forme matricielle,

\[
P^{(m+n)}=P^{(m)}P^{(n)}.
\]

Comme \(P^{(1)}=P\),

\[
P^{(n)}=P^n.
\]

Il faut distinguer l'élément \(p_{ij}^{(n)}\) de la puissance scalaire \((p_{ij})^n\).

## 7. Construction récursive

Si \((\xi_n)\) est une suite i.i.d., si \(X_0\) est indépendante de cette suite et si

\[
X_{n+1}=f(X_n,\xi_{n+1}),
\]

alors le processus ainsi construit est une chaîne de Markov homogène sous les hypothèses du résultat du cours.

## 8. Loi de l'état

On note

\[
\mu_n=(P(X_n=i))_{i\in S}.
\]

La formule des probabilités totales donne

\[
\mu_n=\mu_0P^n.
\]

## 9. Première visite

La probabilité de première visite de \(j\) à partir de \(i\) au temps \(n\) est

\[
f_{ij}^{(n)}=P(X_n=j,X_{n-1}\ne j,\ldots,X_1\ne j\mid X_0=i).
\]

La probabilité totale de visiter \(j\) est

\[
f_{ij}=\sum_{n=1}^{\infty}f_{ij}^{(n)}.
\]

Pour \(i=j\), \(f_{ii}^{(n)}\) est une probabilité de premier retour.

## 10. Accessibilité, communication et classes

L'état \(j\) est accessible depuis \(i\) s'il existe \(n\ge0\) tel que

\[
p_{ij}^{(n)}>0.
\]

Deux états communiquent lorsqu'ils sont accessibles l'un depuis l'autre. La communication est une relation d'équivalence ; ses classes forment une partition de \(S\).

Une chaîne est **irréductible** lorsqu'elle ne possède qu'une seule classe de communication. Une classe est **fermée** lorsqu'il est impossible d'en sortir. Un état est **absorbant** si

\[
p_{ii}=1.
\]

## 11. Récurrence et transience

Un état \(j\) est récurrent si

\[
f_{jj}=1,
\]

et transient si \(f_{jj}<1\).

Le nombre de retours à \(i\), partant de \(i\), est associé à

\[
N(i,i)=\sum_{n=1}^{\infty}\mathbf 1_{\{X_n=i\}}\mathbf 1_{\{X_0=i\}}.
\]

Le cours obtient

\[
E[N(i,i)]=\sum_{n=1}^{\infty}p_{ii}^{(n)}.
\]

Ainsi,

\[
i\text{ récurrent}\iff\sum_{n=1}^{\infty}p_{ii}^{(n)}=\infty,
\]

et la convergence de la série caractérise la transience. La récurrence est une propriété de classe.

## 12. Temps d'atteinte et temps moyen de retour

Le temps de première atteinte de \(j\), partant de \(i\), est

\[
T_{ij}=\min\{n\ge1:X_n=j\mid X_0=i\}.
\]

Alors

\[
f_{ij}^{(n)}=P(T_{ij}=n),
\]

et

\[
\mu_{ij}=E(T_{ij}\mid X_0=i)=\sum_{n=1}^{\infty}nf_{ij}^{(n)}.
\]

Le temps moyen de retour à \(j\) est

\[
\mu_j=E(T_{jj}\mid X_0=j).
\]

Le support établit la décomposition

\[
p_{ij}^{(n)}=\sum_{k=1}^{n}f_{ij}^{(k)}p_{jj}^{(n-k)}.
\]

## 13. Récurrence nulle et positive

Un état récurrent est **récurrent positif** si

\[
\mu_j<\infty,
\]

et **récurrent nul** si

\[
\mu_j=\infty.
\]

Le cours caractérise également les trois catégories par la série \(\sum p_{ii}^{(n)}\) et la limite de \(p_{ii}^{(n)}\) : transience si la série converge ; récurrence nulle si la série diverge mais \(p_{ii}^{(n)}\to0\) ; récurrence positive si la série diverge et que la limite est strictement positive.

## 14. Périodicité et ergodicité

La période d'un état est

\[
d(i)=\operatorname{pgcd}\{n\ge1:p_{ii}^{(n)}>0\}.
\]

Un état est apériodique lorsque \(d(i)=1\). La périodicité est une propriété de classe.

Un état est **ergodique** lorsqu'il est récurrent positif et apériodique. Une chaîne est ergodique lorsque tous ses états le sont. Dans un espace d'états fini, une chaîne irréductible est récurrente positive et, si elle est aussi apériodique, elle est ergodique.

## 15. Distribution stationnaire

Une distribution \(\pi\) est stationnaire si

\[
\pi=\pi P,
\qquad \pi_j\ge0,
\qquad \sum_{j\in S}\pi_j=1.
\]

État par état,

\[
\pi_j=\sum_{i\in S}\pi_i p_{ij}.
\]

Une distribution stationnaire vérifie donc aussi \(\pi=\pi P^n\) pour tout \(n\ge1\).

Le chapitre étudie plusieurs situations : unicité dans certaines chaînes irréductibles, plusieurs distributions lorsque plusieurs classes fermées existent, et absence de distribution stationnaire dans certains cas transitoires.

Si une distribution stationnaire existe, elle donne une masse nulle aux états transitoires ou récurrents nuls.

Si la chaîne est irréductible et récurrente positive, il existe une distribution stationnaire unique et

\[
\boxed{\pi_j=\frac1{\mu_j}}.
\]

Les \(\pi_j\) représentent la proportion de temps passée dans chaque état sur une longue période.

## 16. Distribution limite

La loi de \(X_n\) est

\[
\mu_n=\mu_0P^n.
\]

Le support étudie alors les conditions d'existence de la limite \(\lim_{n\to\infty}\mu_n\). Pour une chaîne ergodique, la matrice \(P^n\) converge vers une matrice dont toutes les lignes sont identiques et la distribution limite coïncide avec la distribution stationnaire.

## 17. Chaînes absorbantes

Le chapitre traite enfin le cas où certaines classes sont absorbantes. Une réorganisation des états permet d'isoler les états transitoires des états absorbants et d'étudier les probabilités et temps d'absorption à partir des blocs de la matrice de transition.

## 18. Synthèse

\[
\text{CMTD}
\rightarrow \text{transition}
\rightarrow \text{Chapman–Kolmogorov}
\rightarrow \text{loi de l'état}
\rightarrow \text{premières visites}
\]

\[
\rightarrow \text{communication}
\rightarrow \text{récurrence/transience}
\rightarrow \text{temps de retour}
\rightarrow \text{périodicité}
\rightarrow \text{stationnarité}
\rightarrow \text{distribution limite/absorption}.
\]