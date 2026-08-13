# Chapitre 2 — Processus de Poisson (PP)

> **Cours de référence :** N. Boussaha, *Processus Aléatoires (2)*, Master Modélisation Stochastique et Prévision en Recherche Opérationnelle (MSPRO), USTHB, 2024–2025.
>
> Cette page appartient à **Course material**. Elle reprend la progression mathématique du support, indépendamment des classes Python et des exemples de programmation qui sont documentés dans les sections séparées du site.

## 1. Introduction

Le processus de Poisson est introduit dans le support comme un processus à temps continu et à valeurs entières positives. Il est utilisé pour représenter dans le temps les instants aléatoires auxquels certains événements se produisent : appels téléphoniques, émissions de particules, pannes, arrivées de clients, etc. Le cours le relie également au cadre des chaînes de Markov à temps continu. fileciteturn467file1L932-L946

## 2. Processus de comptage

Soit \(N(t)\) le nombre d'occurrences observées dans \(]0,t]\). Le processus \((N(t))_{t\ge0}\) est un processus de comptage si :

\[
N(t)\in\mathbb N,\qquad t\ge0,
\]

\[
s<t\quad\Longrightarrow\quad N(s)\le N(t),
\]

et si \(s<t\), alors

\[
N(t)-N(s)
\]

représente le nombre d'occurrences dans l'intervalle \(]s,t]\). fileciteturn467file1L1012-L1021

## 3. Définition du processus de Poisson : première formulation

Un processus de comptage \((N(t))_{t\ge0}\) est un **processus de Poisson de taux \(\lambda>0\)** si :

1. \(N(0)=0\) presque sûrement ;
2. le processus possède des **accroissements indépendants** ;
3. il possède des **accroissements stationnaires** ;
4. pour \(h>0\) suffisamment petit,
   \[
   P(N(h)=1)=\lambda h+o(h),
   \]
   avec \(o(h)/h\to0\) lorsque \(h\to0\) ;
5. \(P(N(h)\ge2)=o(h)\).

Les deux dernières conditions décrivent le comportement sur un intervalle infinitésimal : un événement simple est de probabilité de l'ordre de \(h\), tandis que plusieurs événements sont d'ordre plus petit. fileciteturn467file1L1045-L1062

## 4. Deuxième formulation

Le support donne une formulation équivalente : \(N(0)=0\), les accroissements sont indépendants et, sur un intervalle de longueur \(t\), le nombre d'occurrences suit une loi de Poisson de paramètre \(\lambda t\).

Ainsi, pour \(n\in\mathbb N\),

\[
P\big(N(t+s)-N(s)=n\big)
=
 e^{-\lambda t}\frac{(\lambda t)^n}{n!},
\]

indépendamment de \(s\). Cette propriété implique les accroissements stationnaires. fileciteturn467file1L1140-L1155

## 5. Équivalence des deux définitions

Le cours montre l'implication entre les deux formulations. En posant

\[
p_n(t)=P(N(t)=n),
\]

l'analyse des événements dans un intervalle \([t,t+h]\) conduit d'abord à l'équation

\[
p_0'(t)=-\lambda p_0(t),
\qquad p_0(0)=1,
\]

d'où

\[
p_0(t)=e^{-\lambda t}.
\]

Pour \(n\ge1\), la même décomposition conduit à

\[
p_n'(t)=-\lambda p_n(t)+\lambda p_{n-1}(t),
\qquad p_n(0)=0,
\]

et le support résout cette récurrence différentielle pour obtenir

\[
\boxed{p_n(t)=e^{-\lambda t}\frac{(\lambda t)^n}{n!}}.
\]

La démonstration est construite dans le cours par variation de la constante et récurrence sur \(n\). fileciteturn467file1L1248-L1272 fileciteturn468file1L973-L1009 fileciteturn468file1L1242-L1296

## 6. Temps d'inter-occurrence

Soient

\[
0=\tau_0<\tau_1<\tau_2<\cdots
\]

les instants d'occurrence d'un processus de Poisson de taux \(\lambda\). Le support établit que les temps séparant deux occurrences successives forment une suite de variables aléatoires indépendantes de même loi exponentielle de paramètre \(\lambda\).

Si l'on note une inter-occurrence par \(T\), alors

\[
T\sim\mathrm{Exp}(\lambda).
\]

Le caractère exponentiel et l'indépendance sont obtenus à partir des accroissements indépendants du processus. fileciteturn452file0L1-L3

## 7. Première occurrence conditionnellement à une occurrence unique

Soit \(Y\) l'instant de la première occurrence. Le cours montre que, conditionnellement à l'événement \(N(s)=1\), la variable \(Y\) suit une loi uniforme sur \([0,s]\).

Pour \(0<y<s\), le calcul présenté donne

\[
P(Y\le y\mid N(s)=1)=\frac{y}{s}.
\]

Ainsi,

\[
Y\mid\{N(s)=1\}\sim\mathcal U([0,s]).
\]

fileciteturn452file0L1-L3

## 8. Temps d'occurrence conditionnellement à \(N(s)=k\)

Le résultat suivant généralise le précédent : si l'on conditionne par

\[
N(s)=k,
\]

alors les \(k\) temps d'occurrence observés sur \([0,s]\) ont la loi des statistiques d'ordre obtenues à partir d'un échantillon uniforme sur \([0,s]\).

Autrement dit, les instants d'occurrence conditionnels sont ceux d'un \(k\)-échantillon uniforme ordonné. fileciteturn453file0L1-L2

## 9. Superposition de processus de Poisson

Soient \(N^{(1)}\) et \(N^{(2)}\) deux processus de Poisson indépendants, de taux \(\lambda_1\) et \(\lambda_2\). Le cours considère

\[
N(t)=N^{(1)}(t)+N^{(2)}(t)
\]

et établit que le processus obtenu est encore un processus de Poisson, de taux donné par la somme des taux :

\[
\lambda=\lambda_1+\lambda_2.
\]

Le support laisse la démonstration comme travail dirigé. fileciteturn453file0L1-L2

## 10. Séparation / thinning

À partir d'un processus de Poisson de taux \(\lambda\), le cours attribue indépendamment à chaque occurrence une variable de Bernoulli de paramètre \(p\). Les occurrences sont alors séparées en deux processus :

\[
M(t)=\sum_i X_i,
\qquad
S(t)=\sum_i(1-X_i).
\]

Le support établit que ces deux processus sont des processus de Poisson indépendants, dont les paramètres sont ceux induits par la séparation selon \(p\). fileciteturn453file0L1-L2

## 11. Processus de Poisson non homogène

Le dernier modèle introduit dans le chapitre est le **processus de Poisson non homogène**. Sa différence essentielle avec le cas homogène est que les accroissements ne sont plus stationnaires.

Le support introduit une fonction d'intensité dépendant du temps,

\[
\lambda(t),
\]

et la fonction moyenne

\[
\boxed{m(t)=\int_0^t\lambda(x)\,dx}.
\]

Ce modèle est destiné aux phénomènes dont le taux d'occurrence évolue au cours du temps. fileciteturn453file0L1-L2

Le cours propose notamment comme exercice de partir de l'approximation locale

\[
P(\text{une occurrence sur }[t,t+h[)
=
\lambda(t)h+o(h)
\]

et d'en déduire la loi du nombre d'occurrences sur \([0,t[\). fileciteturn453file0L1-L2

## 12. Synthèse du chapitre

La progression du support est :

\[
\text{processus de comptage}
\rightarrow
\text{définition du PP}
\rightarrow
\text{deuxième caractérisation}
\rightarrow
\text{équivalence}
\]

puis

\[
\text{loi des nombres d'occurrences}
\rightarrow
\text{inter-occurrences exponentielles}
\rightarrow
\text{temps conditionnels}
\rightarrow
\text{superposition}
\rightarrow
\text{séparation}
\rightarrow
\text{PP non homogène}.
\]

Cette page reste volontairement limitée au contenu développé dans le chapitre de Poisson fourni.