# Espérance conditionnelle

## 1. Espace de probabilité fini

Le chapitre utilise le conditionnement dans le cadre discret. OptiFlowX représente explicitement

$$
\Omega,\quad P,
$$

par un `FiniteProbabilitySpace` et une variable aléatoire discrète par un `RandomVariable`.

```python
space = FiniteProbabilitySpace(
    outcomes=[...],
    probabilities=[...],
)
X = space.random_variable([...])
```

L'espérance est

$$
E(X)=\sum_{\omega\in\Omega}X(\omega)P(\{\omega\}).
$$

```python
X.expectation()
```

## 2. Conditionnement par rapport à un événement

Pour $B$ tel que $P(B)>0$ :

$$
E(X\mid B)=\frac{E(X\mathbf 1_B)}{P(B)}.
$$

Et

$$
P(A\mid B)=\frac{P(A\cap B)}{P(B)}.
$$

```python
conditional_expectation_given_event(space, X, B)
conditional_probability_given_event(space, A, B)
```

## 3. Conditionnement par rapport à une variable discrète

Pour $Y$ discrète, $E(X\mid Y)$ est la variable obtenue en donnant sur chaque valeur $y$ la moyenne conditionnelle

$$
E(X\mid Y=y).
$$

```python
space.conditional_expectation_given(X, Y)
```

La partition engendrée par $Y$ rassemble les événements sur lesquels $Y$ prend la même valeur.

```python
Partition.generated_by(Y)
```

## 4. Conditionnement par rapport à une tribu finie

Dans le cadre discret, une tribu finie est représentée par une partition $\mathcal G$. Sur chaque bloc $G$ de la partition,

$$
E(X\mid\mathcal G)=
\frac{1}{P(G)}\sum_{\omega\in G}X(\omega)P(\{\omega\}),
$$

et la valeur est constante sur le bloc.

```python
space.conditional_expectation(X, partition)
```

## 5. Propriétés fondamentales

### Espérance totale

$$
E(E(Y\mid X))=E(Y).
$$

Dans le cadre d'une partition :

```python
space.total_expectation(Y, partition)
```

### Propriété de la tour

Si $\mathcal H\subseteq\mathcal G$ :

$$
E(E(X\mid\mathcal G)\mid\mathcal H)=E(X\mid\mathcal H).
$$

Les partitions sont comparées par raffinement.

```python
space.tower(X, finer, coarser)
```

### Propriété de sortie

Si $Y$ est $\mathcal G$-mesurable :

$$
E(XY\mid\mathcal G)=Y E(X\mid\mathcal G).
$$

```python
space.pull_out(Y, X, partition)
```

## 6. Probabilité conditionnelle comme espérance conditionnelle

Pour un événement $A$,

$$
P(A\mid\mathcal G)=E(\mathbf1_A\mid\mathcal G).
$$

```python
space.conditional_probability(A, partition)
```

## 7. Variance et covariance conditionnelles

Le chapitre donne

$$
Var(X\mid\mathcal G)=E(X^2\mid\mathcal G)-E(X\mid\mathcal G)^2,
$$

et

$$
Cov(X,Y\mid\mathcal G)=E(XY\mid\mathcal G)-E(X\mid\mathcal G)E(Y\mid\mathcal G).
$$

```python
space.conditional_variance(X, partition)
space.conditional_covariance(X, Y, partition)
```

## 8. Formules totales

La variance totale :

$$
Var(X)=E[Var(X\mid\mathcal G)]+Var(E[X\mid\mathcal G]).
$$

La covariance totale :

$$
Cov(X,Y)=E[Cov(X,Y\mid\mathcal G)]
+Cov(E[X\mid\mathcal G],E[Y\mid\mathcal G]).
$$

```python
space.total_variance(X, partition)
space.total_covariance(X, Y, partition)
```

## 9. Projection dans L2

Le chapitre interprète l'espérance conditionnelle comme la projection orthogonale de $X$ sur l'espace des variables aléatoires $\mathcal G$-mesurables dans $L^2$.

Dans l'implémentation discrète :

```python
space.l2_projection(X, partition)
```

## 10. Indépendance

Pour deux partitions finies $\mathcal G_1$ et $\mathcal G_2$, OptiFlowX vérifie l'indépendance par

$$
P(A\cap B)=P(A)P(B)
$$

pour leurs blocs.

Pour deux variables discrètes, les partitions engendrées sont utilisées.

```python
independent_partitions(space, G1, G2)
independent_random_variables(space, X, Y)
```

Le chapitre relie ensuite l'indépendance au fait que connaître une variable indépendante n'apporte pas d'information pour l'espérance conditionnelle.

## 11. Caractérisation

Une caractérisation utilisée dans le chapitre est que, pour tout événement $A$ de la tribu conditionnante,

$$
E[\mathbf1_A X]=E[\mathbf1_A E(X\mid\mathcal G)].
$$

```python
conditional_characterization_error(space, X, partition)
```

Une valeur proche de zéro signifie que cette égalité est vérifiée numériquement sur tous les blocs de la partition.
