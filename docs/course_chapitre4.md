# Chapitre 4 — Espérance conditionnelle

> **Cours de référence :** N. Boussaha, *Processus Aléatoires (4)*, Master Modélisation Stochastique et Prévision en Recherche Opérationnelle (MSPRO), USTHB, 2024–2025.
>
> Cette page appartient à **Course material**. Elle présente les notions et propriétés dans l'ordre du support. Les objets Python correspondants sont traités séparément dans **Package / API**.

## 1. Introduction : estimer avec une information partielle

L'espérance conditionnelle est introduite comme un outil permettant d'estimer une variable aléatoire lorsque seule une information partielle est disponible. Le support la relie notamment aux problèmes de prévision et à l'inférence lorsque certaines données sont non observées ou manquantes. fileciteturn471file0L21-L29

## 2. Conditionnement par rapport à un événement

Sur un espace de probabilité \((\Omega,\mathcal F,P)\), soit \(B\in\mathcal F\) tel que \(P(B)>0\). La probabilité conditionnelle est définie par

\[
P(A\mid B)=\frac{P(A\cap B)}{P(B)}.
\]

Pour une variable aléatoire intégrable \(X\), l'espérance conditionnelle sachant \(B\) est

\[
\boxed{E(X\mid B)=\frac{E(X\mathbf 1_B)}{P(B)}}.
\]

Elle représente la moyenne de \(X\) lorsque l'événement \(B\) est réalisé. fileciteturn471file0L31-L51

### Exemple du cours

Le support considère trois pièces de valeurs 10, 20 et 50 DA. Si \(X\) est la somme obtenue sur pile et si \(B\) est l'événement « exactement deux pièces donnent pile », alors les trois issues de \(B\) ont la même probabilité. Le calcul du cours aboutit à

\[
E(X\mid B)=\frac{160}{3}.
\]

fileciteturn471file0L93-L105 fileciteturn471file0L136-L170

## 3. Conditionnement par rapport à une variable discrète

Soit \(Y\) une variable aléatoire à valeurs dans un espace dénombrable \(E\). On considère

\[
E_0=\{y\in E:P(Y=y)>0\}.
\]

Pour \(y\in E_0\),

\[
E(X\mid Y=y)
=
\frac{E(X\mathbf 1_{\{Y=y\}})}{P(Y=y)}.
\]

Le support définit ensuite l'espérance conditionnelle sachant \(Y\) par une fonction de \(Y\) :

\[
E(X\mid Y)=g(Y),
\]

avec

\[
g(y)=E(X\mid Y=y)
\quad\text{pour }y\in E_0.
\]

La valeur de \(g\) en dehors de \(E_0\) est arbitraire, car cet ensemble a probabilité nulle. fileciteturn471file0L237-L265 fileciteturn471file0L350-L375

Ainsi, contrairement à \(E(X\mid B)\), qui est un nombre, \(E(X\mid Y)\) est une variable aléatoire et une fonction de \(Y\), donc \(\sigma(Y)\)-mesurable. fileciteturn471file0L408-L417

### Exemple du dé

Pour un dé équilibré, le support prend \(X(\omega)=\omega\) et une variable \(Y\) qui indique si le résultat est impair ou pair. On obtient alors une espérance conditionnelle constante sur chacune des deux classes d'information :

\[
E(X\mid Y)=3\mathbf 1_{\{Y=1\}}+4\mathbf 1_{\{Y=0\}}.
\]

fileciteturn471file0L283-L304

## 4. Formulation continue

Dans le cas de variables absolument continues, le cours utilise une densité conditionnelle :

\[
f_{Y\mid X=x}(y)=\frac{f_{X,Y}(x,y)}{f_X(x)}.
\]

Il calcule ensuite

\[
E(Y\mid X=x)=\int y f_{Y\mid X=x}(y)\,dy.
\]

L'exemple développé dans le support conduit à

\[
E(Y\mid X)=X
\quad\text{p.s.}
\]

fileciteturn471file0L430-L453 fileciteturn471file0L455-L530

## 5. Propriétés dans le cas discret

Le support établit notamment :

- \(E(|E(X\mid Y)|)\le E(|X|)\) ;
- la formule de l'espérance totale,
  \[
  E(X)=\sum_y E(X\mid Y=y)P(Y=y);
  \]
- si \(X\) et \(Y\) sont indépendantes, alors, pour les valeurs de \(y\) concernées,
  \[
  E(X\mid Y=y)=E(X);
  \]
- pour une fonction \(h\), on peut conditionner \(h(X,Y)\) à une valeur fixée de \(Y\).

fileciteturn472file0L350-L367

## 6. Caractérisation par la mesurabilité

Pour une variable intégrable \(X\) et une variable discrète \(Y\), le cours donne la caractérisation suivante : \(E(X\mid Y)\) est \(\sigma(Y)\)-mesurable et satisfait, pour tout \(A\in\sigma(Y)\),

\[
\int_A E(X\mid Y)\,dP
=
\int_A X\,dP.
\]

Cette formulation est le point de départ de l'extension à une variable aléatoire arbitraire. fileciteturn472file0L518-L533

## 7. Conditionnement par rapport à une variable arbitraire

Pour une variable aléatoire intégrable \(X\) et une variable arbitraire \(Y\), le cours définit \(E(X\mid Y)\) comme une variable \(\sigma(Y)\)-mesurable vérifiant la même identité d'intégration sur les événements de \(\sigma(Y)\) :

\[
\int_A E(X\mid Y)\,dP
=
\int_A X\,dP,
\qquad A\in\sigma(Y).
\]

Le support précise également que deux variables d'information \(Y\) et \(Y'\) engendrant la même tribu conduisent à la même espérance conditionnelle presque sûrement :

\[
\sigma(Y)=\sigma(Y')
\Longrightarrow
E(X\mid Y)=E(X\mid Y')\quad\text{p.s.}
\]

fileciteturn472file0L592-L607 fileciteturn472file0L678-L755

## 8. Conditionnement par rapport à une tribu

Le chapitre passe ensuite du conditionnement par une variable au conditionnement par une sous-tribu \(\mathcal G\subseteq\mathcal F\).

Pour \(X\in L^1\), la variable

\[
E(X\mid\mathcal G)
\]

est caractérisée par deux propriétés :

1. elle est \(\mathcal G\)-mesurable ;
2. pour tout \(A\in\mathcal G\),
   \[
   \int_AE(X\mid\mathcal G)\,dP
   =
   \int_A X\,dP.
   \]

Dans le cas \(\mathcal G=\sigma(Y)\), on retrouve la notation \(E(X\mid Y)\). fileciteturn472file0L815-L886

## 9. Théorème de caractérisation

Le cours donne une formulation par dualité : il existe une unique variable \(\mathcal G\)-mesurable intégrable \(Y\) telle que, pour toute variable \(Z\) bornée et \(\mathcal G\)-mesurable,

\[
E(ZX)=E(ZY).
\]

Cette variable est précisément

\[
Y=E(X\mid\mathcal G).
\]

En particulier,

\[
E(\mathbf 1_A X)
=
E\big(\mathbf 1_AE(X\mid\mathcal G)\big),
\qquad A\in\mathcal G.
\]

Le support souligne que ce théorème joue un rôle théorique central et que sa démonstration dépasse le cadre du cours. fileciteturn472file0L898-L975

## 10. Propriétés de l'espérance conditionnelle

Le chapitre établit les propriétés suivantes :

### Mesurabilité
Si \(X\) est \(\mathcal G\)-mesurable,

\[
E(X\mid\mathcal G)=X\quad\text{p.s.}
\]

### Linéarité

\[
E(aX+bY\mid\mathcal G)
=
aE(X\mid\mathcal G)+bE(Y\mid\mathcal G).
\]

### Positivité

\[
X\ge0
\Longrightarrow
E(X\mid\mathcal G)\ge0.
\]

### Espérance totale

\[
E(E(X\mid\mathcal G))=E(X).
\]

### Contrôle en valeur absolue

\[
|E(X\mid\mathcal G)|\le E(|X|\mid\mathcal G)\quad\text{p.s.}
\]

et donc

\[
E|E(X\mid\mathcal G)|\le E|X|.
\]

### Monotonie

\[
X\le X'
\Longrightarrow
E(X\mid\mathcal G)\le E(X'\mid\mathcal G)\quad\text{p.s.}
\]

### Indépendance

Si \(X\) est indépendant de \(\mathcal G\), alors

\[
E(X\mid\mathcal G)=E(X)\quad\text{p.s.}
\]

fileciteturn472file1L1022-L1080

## 11. Propriété de sortie d'un facteur mesurable

Si \(Y\) est \(\mathcal G\)-mesurable et que les intégrabilités nécessaires sont satisfaites, alors

\[
\boxed{E(YX\mid\mathcal G)=Y\,E(X\mid\mathcal G).}
\]

Cette propriété est démontrée dans le support à partir du théorème de caractérisation. fileciteturn472file0L1605-L1612 fileciteturn472file0L1774-L1813

## 12. Conditionnement successif : propriété de la tour

Si \(\mathcal G_1\subseteq\mathcal G_2\), alors

\[
\boxed{
E(E(X\mid\mathcal G_2)\mid\mathcal G_1)
=
E(X\mid\mathcal G_1).
}
\]

Le support donne également le cas particulier où \(B\in\mathcal G\) :

\[
E(E(X\mid\mathcal G)\mid B)=E(X\mid B).
\]

fileciteturn472file0L1553-L1577 fileciteturn472file1L1863-L1873

## 13. Indépendance des tribus

Le cours caractérise l'indépendance de deux sous-tribus \(\mathcal G_1\) et \(\mathcal G_2\) par le fait que, pour toute variable intégrable \(X\) mesurable par rapport à \(\mathcal G_2\),

\[
E(X\mid\mathcal G_1)=E(X).
\]

Le support précise aussi qu'en particulier, pour des variables indépendantes \(X\) et \(Y\), on a \(E(X\mid Y)=E(X)\), mais que cette seule égalité ne suffit pas à elle seule pour conclure à l'indépendance. fileciteturn472file1L1817-L1843

## 14. Synthèse du chapitre

La progression du support est :

\[
\text{événement}
\rightarrow
\text{variable discrète}
\rightarrow
\text{variable arbitraire}
\rightarrow
\text{tribu}
\rightarrow
\text{caractérisation}
\rightarrow
\text{propriétés}
\rightarrow
\text{indépendance et propriété de la tour}.
\]

La notion centrale est que \(E(X\mid\mathcal G)\) est la variable intégrable qui représente l'information moyenne sur \(X\) disponible à travers la tribu \(\mathcal G\), avec la propriété de mesurabilité et l'égalité des intégrales sur tous les événements de \(\mathcal G\).

Cette page reste limitée au contenu du chapitre 4 fourni.