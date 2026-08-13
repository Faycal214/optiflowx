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

avec, pour chaque instant \(t\), une variable aléatoire \(X_t\).

Le cours distingue les processus selon que le temps et l'espace des états sont discrets ou continus. La CMTD se situe dans le cas **temps discret / espace des états discret**. Le chapitre utilise cette structure pour représenter des systèmes qui évoluent étape par étape sous l'effet du hasard. fileciteturn467file0L98-L118

## 2. Chaîne de Markov homogène

Soit \((X_n)_{n\in\mathbb N}\) un processus à temps discret et à espace d'états discret \(S\).

### Propriété de Markov

Le processus est une chaîne de Markov lorsque, pour tout \(n\) et tous les états admissibles,

\[
P(X_{n+1}=j\mid X_0=i_0,\ldots,X_{n-1}=i_{n-1},X_n=i)
=
P(X_{n+1}=j\mid X_n=i).
\]

L'idée du chapitre est que, conditionnellement à l'état présent, le passé n'ajoute pas d'information pour prévoir l'état immédiatement suivant. fileciteturn467file0L121-L130

### Homogénéité

La chaîne est homogène lorsque les probabilités de transition ne dépendent pas de l'instant :

\[
P(X_{n+1}=j\mid X_n=i)=P(X_1=j\mid X_0=i).
\]

On note alors

\[
p_{ij}=P(X_{n+1}=j\mid X_n=i).
\]

## 3. Matrice de transition

Les probabilités \(p_{ij}\) sont regroupées dans la matrice

\[
P=(p_{ij})_{i,j\in S}.
\]

Elle est stochastique :

\[
p_{ij}\ge 0,
\qquad
\sum_{j\in S}p_{ij}=1
\quad\text{pour tout }i\in S.
\]

Le support associe aussi à \(P\) un graphe orienté : un état est un sommet et une transition possible est représentée par une arête dont le poids est la probabilité de transition. fileciteturn467file0L133-L170

### Exemple du support : marche aléatoire

Sur \(S=\mathbb Z\), le chapitre considère une chaîne dont les seuls mouvements possibles depuis \(i\) sont vers \(i+1\), vers \(i-1\), ou un maintien en \(i\), avec probabilités \(p,q,r\) telles que

\[
p+q+r=1.
\]

C'est l'exemple de base utilisé pour visualiser une chaîne de Markov sur un espace d'états dénombrable. fileciteturn467file0L172-L185

## 4. Loi initiale et caractérisation

La loi initiale est le vecteur ligne

\[
\mu_0=(P(X_0=i))_{i\in S}.
\]

Une chaîne homogène est entièrement caractérisée par la matrice de transition \(P\) et sa loi initiale \(\mu_0\). Le support obtient notamment la probabilité d'un chemin par factorisation :

\[
P(X_0=i_0,\ldots,X_n=i_n)
=
\mu_0(i_0)p_{i_0i_1}p_{i_1i_2}\cdots p_{i_{n-1}i_n}.
\]

fileciteturn467file0L187-L207

## 5. Transitions en plusieurs étapes

La probabilité de passer de \(i\) à \(j\) en \(n\) transitions est notée

\[
p_{ij}^{(n)}=P(X_{m+n}=j\mid X_m=i).
\]

La matrice correspondante est

\[
P^{(n)}=(p_{ij}^{(n)})_{i,j\in S}.
\]

Le cours rappelle que cette matrice reste stochastique. fileciteturn467file0L253-L272

## 6. Équations de Chapman–Kolmogorov

Pour tous \(m,n\ge 0\),

\[
p_{ij}^{(n+m)}
=
\sum_{k\in S}p_{ik}^{(m)}p_{kj}^{(n)}.
\]

Sous forme matricielle,

\[
P^{(m+n)}=P^{(m)}P^{(n)}.
\]

Comme \(P^{(1)}=P\), on obtient

\[
P^{(n)}=P^n.
\]

Point important du cours : \(p_{ij}^{(n)}\) désigne un élément de la puissance matricielle \(P^n\); ce n'est pas en général la puissance scalaire \((p_{ij})^n\). fileciteturn467file0L317-L385

## 7. Construction récursive

Le chapitre donne le résultat suivant. Si \((\xi_n)\) est une suite i.i.d., si \(X_0\) est indépendant de cette suite et si

\[
X_{n+1}=f(X_n,\xi_{n+1}),
\]

alors le processus ainsi construit est une chaîne de Markov homogène sous les hypothèses du résultat présenté dans le cours. fileciteturn467file0L388-L400

## 8. Loi de l'état

On note

\[
\mu_n=(P(X_n=i))_{i\in S}.
\]

La formule des probabilités totales conduit à

\[
\mu_n=\mu_0P^n.
\]

Cette relation constitue le mécanisme de propagation de la loi initiale au cours du temps. fileciteturn467file0L403-L420

## 9. Probabilités de première visite

La probabilité de première visite de \(j\) à partir de \(i\) au temps \(n\) est

\[
f_{ij}^{(n)}
=
P(X_n=j,\,X_{n-1}\ne j,\ldots,X_1\ne j\mid X_0=i).
\]

La probabilité totale de visiter \(j\) en partant de \(i\) est

\[
f_{ij}=\sum_{n=1}^{\infty}f_{ij}^{(n)}.
\]

Dans le cas \(i=j\), on obtient la probabilité de premier retour à l'état \(i\). fileciteturn467file0L458-L480

## 10. Accessibilité et communication

L'état \(j\) est **accessible** depuis \(i\) lorsqu'il existe \(n\ge0\) tel que

\[
p_{ij}^{(n)}>0.
\]

Les états \(i\) et \(j\) **communiquent** lorsqu'ils sont accessibles l'un depuis l'autre. La relation de communication est une relation d'équivalence et les classes de communication forment une partition de l'espace des états. fileciteturn467file0L505-L525 fileciteturn467file0L653-L685

Une chaîne est **irréductible** lorsqu'elle possède une seule classe de communication.

Une classe est **fermée/absorbante** si, une fois dans cette classe, la chaîne ne peut pas en sortir. Un état \(i\) est absorbant si \(p_{ii}=1\). fileciteturn467file0L688-L700

## 11. Récurrence et transience

Un état \(j\) est **récurrent** si, partant de \(j\), le retour à \(j\) est presque sûr :

\[
f_{jj}=1.
\]

S'il vérifie \(f_{jj}<1\), il est **transient**.

Le cours définit également les chaînes transitoires lorsque tous les états sont transitoires, et les chaînes récurrentes lorsque tous les états sont récurrents. fileciteturn467file0L703-L712

### Nombre moyen de visites

Le nombre de retours à \(i\), en partant de \(i\), est présenté sous la forme

\[
N(i,i)=\sum_{n=1}^{\infty}\mathbf 1_{\{X_n=i\}}\mathbf 1_{\{X_0=i\}}.
\]

Le support obtient

\[
E[N(i,i)]=\sum_{n=1}^{\infty}p_{ii}^{(n)}.
\]

Le critère correspondant est donc :

\[
i\text{ récurrent}
\iff
\sum_{n=1}^{\infty}p_{ii}^{(n)}=+\infty,
\]

et la convergence de cette série caractérise la transience. La récurrence est une propriété de classe. fileciteturn467file0L715-L759 fileciteturn468file0L177-L248

## 12. Temps de première atteinte et temps moyen de retour

Le temps d'atteinte de \(j\), partant de \(i\), est

\[
T_{ij}=\min\{n\ge1:X_n=j\mid X_0=i\}.
\]

Alors

\[
f_{ij}^{(n)}=P(T_{ij}=n)
\]

et le temps moyen d'atteinte est

\[
\mu_{ij}=E(T_{ij}\mid X_0=i)
=\sum_{n=1}^{\infty}n f_{ij}^{(n)}.
\]

Le temps moyen de retour à \(j\) est le cas particulier

\[
\mu_j=E(T_{jj}\mid X_0=j).
\]

fileciteturn467file0L784-L811

Le chapitre établit ensuite la relation fondamentale

\[
p_{ij}^{(n)}
=
\sum_{k=1}^{n}f_{ij}^{(k)}p_{jj}^{(n-k)},
\qquad n\ge1,
\]

en décomposant l'événement \(\{X_n=j\}\) suivant le premier instant auquel \(j\) est atteint. fileciteturn468file0L15-L104

## 13. Récurrence nulle et récurrence positive

Un état récurrent est **récurrent positif** lorsque

\[
\mu_j<\infty,
\]

et **récurrent nul** lorsque

\[
\mu_j=\infty.
\]

Ces deux notions sont des propriétés de classe. fileciteturn468file0L250-L266

Le support donne aussi la caractérisation par la somme des probabilités de retour et la limite de \(p_{ii}^{(n)}\) :

- transient : la somme converge ;
- récurrent nul : la somme diverge et \(p_{ii}^{(n)}\to0\) ;
- récurrent positif : la somme diverge et la limite est strictement positive.

fileciteturn468file0L268-L326

## 14. Périodicité et ergodicité

La période d'un état est définie par

\[
d(i)=\operatorname{pgcd}\{n\ge1:p_{ii}^{(n)}>0\}.
\]

Si aucune de ces probabilités n'est positive, le support fixe \(d(i)=\infty\). Un état est **apériodique** lorsque \(d(i)=1\). La périodicité est une propriété de classe. fileciteturn468file0L358-L375

Un état est **ergodique** lorsqu'il est récurrent positif et apériodique. Une chaîne est ergodique lorsque tous ses états le sont. Pour un espace d'états fini, le cours rappelle notamment qu'une chaîne irréductible est récurrente positive et qu'une chaîne irréductible apériodique est ergodique. fileciteturn468file0L377-L390

## 15. Distribution stationnaire

Une distribution \(\pi\) est stationnaire lorsque

\[
\pi=\pi P,
\qquad
\pi_j\ge0,
\qquad
\sum_{j\in S}\pi_j=1.
\]

État par état :

\[
\pi_j=\sum_{i\in S}\pi_i p_{ij}.
\]

Une distribution stationnaire reste inchangée après chaque transition, donc

\[
\pi=\pi P^n,
\qquad n\ge1.
\]

fileciteturn468file0L393-L412

Le support insiste sur plusieurs situations : une distribution stationnaire peut être unique, une chaîne périodique peut néanmoins posséder une distribution stationnaire unique, plusieurs classes fermées peuvent conduire à plusieurs distributions stationnaires, et une chaîne transitoire peut n'en posséder aucune. fileciteturn468file0L414-L465

Si une distribution stationnaire existe, elle ne charge aucun état transitoire ou récurrent nul. fileciteturn468file0L538-L577

Pour une chaîne irréductible récurrente positive, le cours établit l'existence et l'unicité de la distribution stationnaire et la relation

\[
\boxed{\pi_j=\frac1{\mu_j}}.
\]

La distribution stationnaire est interprétée comme la proportion de temps passée dans chaque état sur une longue période. fileciteturn468file0L592-L650

## 16. Distribution limite

Le chapitre distingue clairement **distribution stationnaire** et **distribution limite**.

Une chaîne admet une distribution limite si, pour tous \(i,j\),

\[
\lim_{n\to\infty}p_{ij}^{(n)}=\pi_j.
\]

Pour une chaîne ergodique, le cours établit l'existence de cette limite et l'identifie à la distribution stationnaire. Il rappelle aussi qu'une chaîne peut avoir une distribution stationnaire unique mais ne pas avoir de distribution limite lorsqu'elle est périodique. fileciteturn468file0L698-L728 fileciteturn468file0L890-L908

Lorsque la chaîne possède une unique classe fermée ergodique et que les autres états sont transitoires, la masse limite se concentre sur cette classe ; les états transitoires reçoivent une masse limite nulle.

## 17. Probabilités d'absorption

Dans la partie asymptotique, le support considère des états transitoires et des classes fermées ergodiques \(C_k\). La probabilité d'absorption par une classe \(C_k\), en partant d'un état transitoire \(i\), est obtenue à partir des probabilités d'atteindre cette classe.

Lorsque l'ensemble des états transitoires \(T\) est fini, la probabilité d'absorption par \(C_k\) est donnée comme solution du système

\[
\pi_i(C_k)
=
\sum_{j\in C_k}p_{ij}
+
\sum_{j\in T}p_{ij}\pi_j(C_k).
\]

Le support présente cette construction comme l'outil permettant ensuite de caractériser le comportement asymptotique des probabilités de transition lorsqu'une classe ergodique est atteignable depuis les états transitoires.

## 18. Synthèse du chapitre

La progression mathématique du chapitre est :

\[
\text{processus aléatoire}
\rightarrow
\text{propriété de Markov}
\rightarrow
P,\mu_0
\rightarrow
P^n,\mu_n
\rightarrow
\text{premières visites}
\rightarrow
\text{communication}
\]

puis

\[
\text{récurrence/transience}
\rightarrow
\text{temps de retour}
\rightarrow
\text{périodicité}
\rightarrow
\text{ergodicité}
\rightarrow
\text{stationnarité}
\rightarrow
\text{distribution limite}
\rightarrow
\text{absorption}.
\]

Cette page reste limitée aux notions développées dans le support du chapitre 1.