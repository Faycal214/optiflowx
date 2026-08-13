# Processus de naissance et de mort

## 1. Structure

Le processus de naissance et de mort est un cas de CMTC à états dans $\mathbb N$ où, depuis l'état $k$, seuls les sauts

$$k\to k+1\quad\text{et}\quad k\to k-1$$

sont possibles.

On note $\lambda_k$ le taux de naissance et $\mu_k$ le taux de mort.

La matrice génératrice possède donc les termes

$$
q_{k,k+1}=\lambda_k,\qquad q_{k,k-1}=\mu_k,
$$

et

$$
q_{kk}=-(\lambda_k+\mu_k).
$$

```python
from optiflowx.stochastic import BirthDeathProcess
process = BirthDeathProcess(...)
```

## 2. Générateur et chaîne des sauts

Pour un modèle fini :

```python
process.generator_matrix()
process.jump_chain_matrix()
process.to_ctmc()
```

La chaîne des sauts utilise

$$
\tilde p_{k,k+1}=\frac{\lambda_k}{\lambda_k+\mu_k},
\qquad
\tilde p_{k,k-1}=\frac{\mu_k}{\lambda_k+\mu_k}
$$

lorsque le taux total est positif.

## 3. Équations de Kolmogorov

Le vecteur des probabilités $p_k(t)=P(X_t=k)$ suit les équations de naissance et de mort obtenues à partir de la matrice $Q$.

Pour un calcul fini, `kolmogorov_derivative(p)` retourne le vecteur $p'(t)=p(t)Q$ sous la convention ligne utilisée dans le package.

## 4. Distribution stationnaire

Dans le cas où une distribution stationnaire existe et sous les conditions du chapitre, elle s'écrit à partir du produit

$$
\rho_0=1,
\qquad
\rho_k=\rho_{k-1}\frac{\lambda_{k-1}}{\mu_k},
$$

puis par normalisation lorsque la somme des poids est finie.

```python
process.stationary_weights(n_terms)
```

Pour un espace fini, `stationary_distribution()` résout la relation stationnaire de la CMTC correspondante.

## 5. Cas particuliers du chapitre

### Immigration pure

Avec

$$\lambda_k=\lambda,\qquad \mu_k=0,$$

le processus est un processus de Poisson dans le sens présenté dans le chapitre.

```python
BirthDeathProcess.pure_immigration_probability(n, t, rate=lam)
```

et

$$
P(X_t=n)=e^{-\lambda t}\frac{(\lambda t)^n}{n!}
$$

pour le cas correspondant.

### Naissance pure

Avec

$$\lambda_k=k\lambda,
$$

et sans morts, le chapitre donne les probabilités explicites du processus de naissance pure.

```python
BirthDeathProcess.pure_birth_probability(n, t, rate=lam)
```

### Mort pure

Avec

$$\mu_k=k\mu,
$$

et sans naissances, le chapitre donne une loi binomiale pour la population restante à partir d'une population initiale donnée.

```python
BirthDeathProcess.pure_death_probability(...)
```

### Modèle linéaire avec immigration

Le chapitre étudie notamment des taux de la forme

$$
\lambda_n=n\lambda+\alpha,
\qquad
\mu_n=n\mu.
$$

```python
BirthDeathProcess.linear(
    birth_rate=lam,
    death_rate=mu,
    immigration=alpha,
)
```

## 6. Explosion

Pour un processus de naissance pure, le chapitre relie la non-explosion à la série des inverses des taux de naissance. OptiFlowX expose la somme partielle calculable :

```python
process.pure_birth_reciprocal_rate_sum(n_terms)
```

Cette fonction ne prétend pas décider la convergence d'une série infinie à partir d'une troncature finie.

## 7. Composants principaux

| Mathématique | API |
|---|---|
| Taux $\lambda_k$ | `birth_rate` |
| Taux $\mu_k$ | `death_rate` |
| Générateur | `generator_matrix` |
| Chaîne des sauts | `jump_chain_matrix` |
| CMTC associée | `to_ctmc` |
| Équation de Kolmogorov | `kolmogorov_derivative` |
| Poids stationnaires | `stationary_weights` |
| Immigration pure | `pure_immigration_probability` |
| Naissance pure | `pure_birth_probability` |
| Mort pure | `pure_death_probability` |
| Critère d'explosion, somme partielle | `pure_birth_reciprocal_rate_sum` |
