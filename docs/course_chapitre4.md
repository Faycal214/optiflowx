# Chapitre 4 — Espérance conditionnelle

> **Cours de référence :** N. Boussaha, *Processus Aléatoires (4)*, Master Modélisation Stochastique et Prévision en Recherche Opérationnelle (MSPRO), USTHB, 2024–2025.
>
> Cette page appartient à **Course material**. Elle présente les notions et propriétés dans l'ordre du support. Les objets Python correspondants sont traités séparément dans **Package / API**.

## 1. Introduction

L'espérance conditionnelle est introduite comme un outil d'estimation lorsqu'une information partielle est disponible, notamment en prévision et en présence de données non observées ou manquantes.

## 2. Conditionnement par rapport à un événement

Pour \(B\in\mathcal F\) tel que \(P(B)>0\),

\[
P(A\mid B)=\frac{P(A\cap B)}{P(B)}.
\]

Pour \(X\in L^1\),

\[
\boxed{E(X\mid B)=\frac{E(X\mathbf 1_B)}{P(B)}}.
\]

### Exemple du support

Trois pièces de valeurs 10, 20 et 50 DA sont lancées. Si \(X\) est le montant total obtenu sur pile et \(B\) l'événement « exactement deux pièces montrent pile », alors

\[
E(X\mid B)=\frac{160}{3}.
\]

## 3. Conditionnement par rapport à une variable discrète

Soit \(Y\) à valeurs dans un espace dénombrable \(E\), et

\[
E_0=\{y\in E:P(Y=y)>0\}.
\]

Pour \(y\in E_0\),

\[
E(X\mid Y=y)=\frac{E(X\mathbf1_{\{Y=y\}})}{P(Y=y)}.
\]

On définit ensuite

\[
E(X\mid Y)=g(Y),
\]

où \(g(y)=E(X\mid Y=y)\) sur \(E_0\). La valeur de \(g\) hors de \(E_0\) est sans conséquence car cet ensemble a probabilité nulle.

Ainsi \(E(X\mid Y)\) est une variable aléatoire et une fonction de \(Y\), donc \(\sigma(Y)\)-mesurable.

### Exemple du dé

Pour un dé équilibré, si \(X(\omega)=\omega\) et si \(Y\) indique si le résultat est impair ou pair, le cours obtient

\[
E(X\mid Y)=3\mathbf1_{\{Y=1\}}+4\mathbf1_{\{Y=0\}}.
\]

## 4. Cas absolument continu

Lorsque \(X\) et \(Y\) possèdent une densité conjointe,

\[
f_{Y\mid X=x}(y)=\frac{f_{X,Y}(x,y)}{f_X(x)}.
\]

Alors

\[
E(Y\mid X=x)=\int y f_{Y\mid X=x}(y)\,dy.
\]

Dans l'exemple développé par le support, cette procédure conduit à

\[
E(Y\mid X)=X\quad\text{p.s.}
\]

## 5. Propriétés dans le cas discret

Le support établit notamment

\[
E|E(X\mid Y)|\le E|X|,
\]

la formule de l'espérance totale

\[
E(X)=\sum_yE(X\mid Y=y)P(Y=y),
\]

et, en cas d'indépendance,

\[
E(X\mid Y=y)=E(X).
\]

Pour une fonction \(h\), la valeur de \(Y\) peut être remplacée par la constante correspondante à l'intérieur du conditionnement.

## 6. Caractérisation par \(\sigma(Y)\)

Pour \(X\in L^1\) et \(Y\) discrète, \(E(X\mid Y)\) est l'unique objet, à un ensemble de probabilité nulle près, qui est \(\sigma(Y)\)-mesurable et vérifie

\[
\int_A E(X\mid Y)\,dP
=
\int_A X\,dP,
\qquad A\in\sigma(Y).
\]

Cette formulation sert de transition vers le conditionnement par une variable arbitraire puis par une tribu.

## 7. Variable aléatoire arbitraire

Pour \(X\in L^1\) et une variable aléatoire arbitraire \(Y\), l'espérance conditionnelle est définie par la même propriété : elle est \(\sigma(Y)\)-mesurable et

\[
\int_A E(X\mid Y)\,dP
=
\int_A X\,dP,
\qquad A\in\sigma(Y).
\]

Le cours souligne que la tribu d'information est la notion pertinente. En particulier,

\[
\sigma(Y)=\sigma(Y')
\Longrightarrow
E(X\mid Y)=E(X\mid Y')\quad\text{p.s.}
\]

## 8. Conditionnement par rapport à une tribu

Soit \(\mathcal G\subseteq\mathcal F\) une sous-tribu. Pour \(X\in L^1\), l'espérance conditionnelle \(E(X\mid\mathcal G)\) est une variable \(\mathcal G\)-mesurable telle que

\[
\int_AE(X\mid\mathcal G)\,dP
=
\int_AX\,dP,
\qquad A\in\mathcal G.
\]

Dans le cas \(\mathcal G=\sigma(Y)\), on retrouve la notation \(E(X\mid Y)\).

## 9. Théorème de caractérisation

Il existe une unique variable \(Y\in L^1(\Omega,\mathcal G,P)\) telle que, pour toute variable \(Z\) bornée et \(\mathcal G\)-mesurable,

\[
E(ZX)=E(ZY).
\]

Cette variable est

\[
Y=E(X\mid\mathcal G).
\]

En particulier,

\[
E(\mathbf1_AX)=E\big(\mathbf1_AE(X\mid\mathcal G)\big),
\qquad A\in\mathcal G.
\]

La démonstration d'existence du théorème dépasse le cadre du cours.

## 10. Propriétés fondamentales

Si \(X\) est \(\mathcal G\)-mesurable,

\[
E(X\mid\mathcal G)=X\quad\text{p.s.}
\]

Linéarité :

\[
E(aX+bY\mid\mathcal G)
=aE(X\mid\mathcal G)+bE(Y\mid\mathcal G).
\]

Positivité :

\[
X\ge0\Rightarrow E(X\mid\mathcal G)\ge0.
\]

Espérance totale :

\[
E(E(X\mid\mathcal G))=E(X).
\]

Contrôle absolu :

\[
|E(X\mid\mathcal G)|\le E(|X|\mid\mathcal G),
\]

et donc \(E|E(X\mid\mathcal G)|\le E|X|\).

Monotonie :

\[
X\le X'\Rightarrow E(X\mid\mathcal G)\le E(X'\mid\mathcal G)\quad\text{p.s.}
\]

Indépendance : si \(X\) est indépendant de \(\mathcal G\),

\[
E(X\mid\mathcal G)=E(X)\quad\text{p.s.}
\]

## 11. Facteur mesurable

Si \(Y\) est \(\mathcal G\)-mesurable et que les intégrabilités nécessaires sont satisfaites,

\[
\boxed{E(YX\mid\mathcal G)=Y\,E(X\mid\mathcal G).}
\]

## 12. Conditionnement successif

Si \(\mathcal G_1\subseteq\mathcal G_2\), alors

\[
\boxed{
E(E(X\mid\mathcal G_2)\mid\mathcal G_1)=E(X\mid\mathcal G_1).
}
\]

Si \(B\in\mathcal G\), le cours donne également

\[
E(E(X\mid\mathcal G)\mid B)=E(X\mid B).
\]

## 13. Indépendance des tribus

Deux sous-tribus \(\mathcal G_1\) et \(\mathcal G_2\) sont indépendantes si et seulement si, pour toute variable intégrable mesurable par rapport à \(\mathcal G_2\),

\[
E(X\mid\mathcal G_1)=E(X).
\]

Pour des variables indépendantes \(X\) et \(Y\), on retrouve en particulier

\[
E(X\mid Y)=E(X).
\]

Le cours souligne toutefois que cette dernière égalité, prise isolément, ne suffit pas à établir l'indépendance.

## 14. Synthèse

\[
\text{événement}
\rightarrow\text{variable discrète}
\rightarrow\text{variable arbitraire}
\rightarrow\text{tribu}
\]

\[
\rightarrow\text{caractérisation}
\rightarrow\text{propriétés}
\rightarrow\text{indépendance}
\rightarrow\text{propriété de la tour}.
\]