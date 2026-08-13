# Processus de naissance et de mort

Cette page suit la partie **Processus de naissance et de mort** du Chapitre 3. Le processus est présenté dans le PDF comme une classe particulière de CMTC ; la documentation conserve donc cette relation.

## 1. Définition

L'espace d'états est $\mathbb N$. Depuis l'état $k$, seuls deux sauts sont possibles :

$$
k\longrightarrow k+1
\qquad\text{et}\qquad
k\longrightarrow k-1.
$$

On note $\lambda_k$ le taux de naissance et $\mu_k$ le taux de mort. La matrice génératrice a alors la forme

$$
q_{k,k+1}=\lambda_k,
\qquad
q_{k,k-1}=\mu_k,
\qquad
q_{k,k}=-(\lambda_k+\mu_k),
$$

avec les ajustements nécessaires aux frontières.

```python
from optiflowx.stochastic import BirthDeathProcess

process = BirthDeathProcess(
    birth_rates=lambda k: 0.4 * k,
    death_rates=lambda k: 0.2 * k,
)
```

## 2. Chaîne des sauts

Lorsque $\lambda_k+\mu_k>0$, les probabilités de la chaîne des sauts sont

$$
\widetilde p_{k,k+1}
=\frac{\lambda_k}{\lambda_k+\mu_k},
\qquad
\widetilde p_{k,k-1}
=\frac{\mu_k}{\lambda_k+\mu_k}.
$$

Pour un modèle fini :

```python
process.jump_chain_matrix()
process.to_ctmc()
```

## 3. Équations de Kolmogorov

Si

$$
p_k(t)=P(X_t=k),
$$

les probabilités vérifient les équations de naissance et de mort obtenues à partir de $Q$. Pour un modèle fini, `kolmogorov_derivative` représente le vecteur $p'(t)=p(t)Q$ selon la convention ligne du package.

```python
p_derivative = process.kolmogorov_derivative(probabilities)
```

## 4. Distribution stationnaire

Le chapitre construit les poids

$$
\rho_0=1,
\qquad
\rho_k
=\rho_{k-1}\frac{\lambda_{k-1}}{\mu_k}.
$$

Lorsque la normalisation est possible, ces poids donnent la distribution stationnaire.

```python
weights = process.stationary_weights(n_terms=20)
```

Pour un modèle fini, la distribution stationnaire est obtenue via la CMTC correspondante :

```python
pi = process.stationary_distribution()
```

## 5. Croissance pure par immigration

Le chapitre considère

$$
\lambda_k=\alpha>0,
\qquad
\mu_k=0.
$$

Il retrouve alors le processus de Poisson de paramètre $\alpha$ et

$$
P(X_t=n)=e^{-\alpha t}\frac{(\alpha t)^n}{n!}.
$$

```python
BirthDeathProcess.pure_immigration_probability(
    n=3,
    t=2.0,
    rate=alpha,
)
```

## 6. Croissance pure par naissance

Le chapitre traite le cas

$$
\lambda_n=n\lambda,
\qquad
\mu_n=0.
$$

La probabilité correspondante est exposée par :

```python
BirthDeathProcess.pure_birth_probability(
    n=4,
    t=2.0,
    rate=lam,
)
```

Le même chapitre relie ensuite la non-explosion du processus de naissance pure au comportement de la série des inverses des taux de naissance. Le package retourne seulement une somme partielle :

```python
process.pure_birth_reciprocal_rate_sum(n_terms=100)
```

Une troncature numérique n'est pas présentée comme une preuve de convergence de la série infinie.

## 7. Mort pure

Dans le cas

$$
\lambda_n=0,
\qquad
\mu_n=n\mu,
$$

le chapitre donne une loi binomiale pour la population restante à partir d'une population initiale donnée.

```python
BirthDeathProcess.pure_death_probability(
    n=3,
    t=2.0,
    initial_population=10,
    rate=mu,
)
```

## 8. Taux linéaires

Le chapitre étudie notamment les taux

$$
\lambda_n=n\lambda+\alpha,
\qquad
\mu_n=n\mu+\beta,
$$

où $\alpha$ représente l'immigration et $\beta$ l'émigration.

```python
process = BirthDeathProcess.linear(
    birth_rate=lam,
    death_rate=mu,
    immigration=alpha,
    emigration=beta,
)
```

Le cas particulier de croissance pure par immigration est explicitement relié au processus de Poisson dans le chapitre.

## 9. Correspondance cours → package

| Objet du chapitre | Composant OptiFlowX |
|---|---|
| $\lambda_k$ | `birth_rate` |
| $\mu_k$ | `death_rate` |
| Générateur $Q$ | `generator_matrix` |
| Chaîne des sauts | `jump_chain_matrix` |
| CMTC associée | `to_ctmc` |
| Équations de Kolmogorov | `kolmogorov_derivative` |
| Poids stationnaires | `stationary_weights` |
| Immigration pure | `pure_immigration_probability` |
| Naissance pure | `pure_birth_probability` |
| Mort pure | `pure_death_probability` |
| Série de l'explosion | `pure_birth_reciprocal_rate_sum` |
