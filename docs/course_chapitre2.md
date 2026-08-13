# Chapitre 2 — Processus de Poisson (PP)

> **Cours de référence :** N. Boussaha, *Processus Aléatoires (2)*, Master Modélisation Stochastique et Prévision en Recherche Opérationnelle (MSPRO), USTHB, 2024–2025.
>
> Cette page appartient à **Course material**. Elle reprend la progression mathématique du support, indépendamment des classes Python et des exemples de programmation documentés ailleurs.

## 1. Introduction

Le processus de Poisson est introduit comme un processus à temps continu et à valeurs entières positives, utilisé pour décrire les instants auxquels des événements aléatoires se produisent : appels, arrivées de clients, pannes, émissions de particules, etc.

## 2. Processus de comptage

Soit \(N(t)\) le nombre d'occurrences observées dans \(]0,t]\). Un processus \((N(t))_{t\ge0}\) est un processus de comptage si

\[
N(t)\in\mathbb N,
\qquad s<t\Rightarrow N(s)\le N(t),
\]

et si \(N(t)-N(s)\) représente le nombre d'occurrences sur \(]s,t]\).

## 3. Première définition du processus de Poisson

Un processus de comptage est un processus de Poisson de taux \(\lambda>0\) si :

- \(N(0)=0\) presque sûrement ;
- les accroissements sont indépendants ;
- les accroissements sont stationnaires ;
- \(P(N(h)=1)=\lambda h+o(h)\) ;
- \(P(N(h)\ge2)=o(h)\).

Ici \(o(h)/h\to0\) lorsque \(h\to0\).

## 4. Deuxième définition

Le support donne la formulation équivalente : \(N(0)=0\), les accroissements sont indépendants et, pour toute longueur \(t\),

\[
P(N(t+s)-N(s)=n)=e^{-\lambda t}\frac{(\lambda t)^n}{n!},
\qquad n\in\mathbb N.
\]

Cette loi ne dépend pas de \(s\), ce qui donne les accroissements stationnaires.

## 5. Équivalence des définitions

En posant

\[
p_n(t)=P(N(t)=n),
\]

l'analyse sur \([t,t+h]\) donne

\[
p_0'(t)=-\lambda p_0(t),\qquad p_0(0)=1,
\]

et, pour \(n\ge1\),

\[
p_n'(t)=-\lambda p_n(t)+\lambda p_{n-1}(t),
\qquad p_n(0)=0.
\]

La résolution conduit à

\[
\boxed{P(N(t)=n)=e^{-\lambda t}\frac{(\lambda t)^n}{n!}}.
\]

Donc \(N(t)\sim\mathcal P(\lambda t)\) et

\[
N(t+s)-N(s)\sim\mathcal P(\lambda t).
\]

## 6. Temps inter-occurrences

Si

\[
0=\tau_0<\tau_1<\tau_2<\cdots
\]

sont les temps d'occurrence, les inter-arrivées

\[
T_n=\tau_n-\tau_{n-1}
\]

forment une suite i.i.d. de loi exponentielle de paramètre \(\lambda\) :

\[
T_n\sim\mathrm{Exp}(\lambda).
\]

## 7. Temps d'occurrence conditionnels

Conditionnellement à \(N(s)=1\), l'instant de l'unique occurrence dans \([0,s]\) est uniforme :

\[
T_1\mid\{N(s)=1\}\sim\mathcal U([0,s]).
\]

Plus généralement, conditionnellement à \(N(s)=k\), les \(k\) temps d'occurrence sur \([0,s]\) ont la loi des statistiques d'ordre d'un échantillon de \(k\) variables uniformes sur \([0,s]\).

## 8. Superposition

Si \(N^{(1)}\) et \(N^{(2)}\) sont indépendants et de taux \(\lambda_1\) et \(\lambda_2\), alors

\[
N(t)=N^{(1)}(t)+N^{(2)}(t)
\]

est encore un processus de Poisson, de taux

\[
\boxed{\lambda_1+\lambda_2}.
\]

## 9. Séparation / thinning

À partir d'un processus de Poisson de taux \(\lambda\), chaque occurrence peut être affectée indépendamment à un type 1 avec probabilité \(p\), ou au type 2 avec probabilité \(1-p\). Les deux processus obtenus sont indépendants et ont pour taux

\[
\lambda p
\qquad\text{et}\qquad
\lambda(1-p).
\]

## 10. Processus de Poisson non homogène

Dans le cas non homogène, le taux dépend du temps : \(\lambda=\lambda(t)\). Les accroissements ne sont donc plus stationnaires. On introduit la fonction moyenne cumulée

\[
\boxed{m(t)=\int_0^t\lambda(u)\,du}.
\]

Localement,

\[
P(\text{une occurrence dans }[t,t+h[)=\lambda(t)h+o(h).
\]

Ce modèle convient aux phénomènes dont le rythme d'occurrence varie avec le temps.

## 11. Résultats à retenir

\[
N(t)\sim\mathcal P(\lambda t),
\qquad
T_n\sim\mathrm{Exp}(\lambda).
\]

Les accroissements sur des intervalles disjoints sont indépendants ; dans le cas homogène, leur loi dépend seulement de la longueur des intervalles. Conditionnellement à un nombre fixé d'occurrences, les instants sont distribués comme des statistiques d'ordre uniformes.

## 12. Synthèse

\[
\text{comptage}
\rightarrow \text{définition du PP}
\rightarrow \text{loi de Poisson}
\rightarrow \text{inter-occurrences}
\]

\[
\rightarrow \text{temps conditionnels}
\rightarrow \text{superposition}
\rightarrow \text{séparation}
\rightarrow \text{PP non homogène}.
\]