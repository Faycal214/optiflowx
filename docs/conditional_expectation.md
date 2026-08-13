# Espérance conditionnelle

Cette page suit le **Chapitre 4 — Espérance conditionnelle**. Le package utilise une représentation finie et discrète pour rendre les définitions du cours calculables directement.

## 1. Conditionnement par rapport à un événement

Soit $B$ un événement tel que $P(B)>0$. Le chapitre définit

$$
P(A\mid B)=\frac{P(A\cap B)}{P(B)}
$$

et, pour une variable aléatoire intégrable $X$,

$$
E(X\mid B)=\frac{E(X\mathbf 1_B)}{P(B)}.
$$

Le cours interprète cette quantité comme la valeur moyenne de $X$ lorsque $B$ est réalisé. fileciteturn344file1L106-L138

Dans OptiFlowX :

```python
space.conditional_probability_given_event(A, B)
space.conditional_expectation_given_event(X, B)
```

## 2. Espace de probabilité fini et variable aléatoire

Le package représente explicitement les issues et leurs probabilités :

```python
space = FiniteProbabilitySpace(
    outcomes=[...],
    probabilities=[...],
)
X = space.random_variable([...], name="X")
```

L'espérance est calculée par la somme pondérée des valeurs :

$$
E(X)=\sum_{\omega\in\Omega}X(\omega)P(\{\omega\}).
$$

```python
X.expectation()
```

## 3. Conditionnement par rapport à une tribu

Pour une sous-tribu $\mathcal G$, le chapitre définit $E(X\mid\mathcal G)$ comme une variable $\mathcal G$-mesurable vérifiant la propriété intégrale de caractérisation :

$$
\int_A E(X\mid\mathcal G)\,dP
=
\int_A X\,dP,
\qquad A\in\mathcal G.
$$

Le théorème de caractérisation présenté dans le cours donne également la formulation avec les variables $\mathcal G$-mesurables bornées. fileciteturn344file2L170-L203

Dans notre représentation finie, une tribu finie est représentée par une `Partition`. Sur chaque bloc $G$ de probabilité positive :

$$
E(X\mid\mathcal G)(\omega)
=
\frac{\sum_{u\in G}X(u)P(\{u\})}{P(G)},
\qquad \omega\in G.
$$

```python
G = space.partition([
    {"omega_1", "omega_2"},
    {"omega_3", "omega_4"},
])
conditioned = space.conditional_expectation(X, G)
```

## 4. Conditionnement par rapport à une variable

Lorsque la tribu est engendrée par une variable $Y$, le chapitre écrit

$$
E(X\mid\sigma(Y))=E(X\mid Y).
$$

De même,

$$
P(A\mid\mathcal G)=E(\mathbf 1_A\mid\mathcal G).
$$

fileciteturn344file5L409-L422

OptiFlowX construit la partition engendrée par $Y$ en regroupant les issues ayant la même valeur de $Y$ :

```python
G_Y = Partition.generated_by(Y)
E_X_given_Y = space.conditional_expectation_given(X, Y)
```

## 5. Propriétés fondamentales

Le chapitre établit notamment :

### Mesurabilité

Si $X$ est $\mathcal G$-mesurable :

$$
E(X\mid\mathcal G)=X\quad\text{p.s.}
$$

### Linéarité

$$
E(aX+bY\mid\mathcal G)
=aE(X\mid\mathcal G)+bE(Y\mid\mathcal G).
$$

### Positivité

Si $X\geq0$, alors

$$
E(X\mid\mathcal G)\geq0.
$$

### Espérance totale

$$
E(E(X\mid\mathcal G))=E(X).
$$

Ces propriétés sont explicitement regroupées dans le chapitre. fileciteturn344file2L204-L212

Dans OptiFlowX, l'identité d'espérance totale peut être vérifiée par :

```python
space.total_expectation(X, G)
```

## 6. Propriété de la tour

Si $\mathcal G_1\subseteq\mathcal G_2$ :

$$
E(E(X\mid\mathcal G_2)\mid\mathcal G_1)
=
E(X\mid\mathcal G_1).
$$

fileciteturn344file4L318-L330

Dans la représentation par partitions, `finer` représente la tribu contenant davantage d'information et `coarser` la tribu plus petite :

```python
space.tower(X, finer, coarser)
```

## 7. Propriété de sortie

Si $Y$ est $\mathcal G$-mesurable :

$$
E(YX\mid\mathcal G)
=
Y E(X\mid\mathcal G).
$$

fileciteturn344file3L237-L294

```python
residual = space.pull_out(Y, X, G)
```

Un résidu nul signifie que l'identité est vérifiée numériquement sur la représentation finie.

## 8. Probabilité conditionnelle comme espérance

Le cours identifie

$$
P(A\mid\mathcal G)=E(\mathbf 1_A\mid\mathcal G).
$$

```python
space.conditional_probability(A, G)
```

## 9. Variance et covariance conditionnelles

Dans la partie consacrée aux variables de carré intégrable, le package représente les décompositions conditionnelles directement :

$$
Var(X\mid\mathcal G)
=E(X^2\mid\mathcal G)-E(X\mid\mathcal G)^2,
$$

et

$$
Cov(X,Y\mid\mathcal G)
=E(XY\mid\mathcal G)
-E(X\mid\mathcal G)E(Y\mid\mathcal G).
$$

```python
space.conditional_variance(X, G)
space.conditional_covariance(X, Y, G)
```

## 10. Formules totales

Le package expose les décompositions correspondantes :

$$
Var(X)
=E[Var(X\mid\mathcal G)]
+Var(E[X\mid\mathcal G]),
$$

et

$$
Cov(X,Y)
=E[Cov(X,Y\mid\mathcal G)]
+Cov(E[X\mid\mathcal G],E[Y\mid\mathcal G]).
$$

```python
space.total_variance(X, G)
space.total_covariance(X, Y, G)
```

## 11. Projection dans $L^2$

Pour les variables de carré intégrable, le chapitre donne également l'interprétation de l'espérance conditionnelle comme projection sur l'espace des variables $\mathcal G$-mesurables.

```python
space.l2_projection(X, G)
```

## 12. Correspondance cours → package

| Objet du chapitre | Composant OptiFlowX |
|---|---|
| Espace fini | `FiniteProbabilitySpace` |
| Variable aléatoire | `RandomVariable` |
| Partition / tribu finie | `Partition` |
| $P(A)$ | `probability` |
| $P(A\mid B)$ | `conditional_probability_given_event` |
| $E(X\mid B)$ | `conditional_expectation_given_event` |
| $E(X)$ | `RandomVariable.expectation` |
| $E(X\mid\mathcal G)$ | `conditional_expectation` |
| $E(X\mid Y)$ | `conditional_expectation_given` |
| $P(A\mid\mathcal G)$ | `conditional_probability` |
| Espérance totale | `total_expectation` |
| Propriété de la tour | `tower` |
| Propriété de sortie | `pull_out` |
| Variance conditionnelle | `conditional_variance` |
| Covariance conditionnelle | `conditional_covariance` |
| Formule de variance totale | `total_variance` |
| Formule de covariance totale | `total_covariance` |
| Projection $L^2$ | `l2_projection` |
