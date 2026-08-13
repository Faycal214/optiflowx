# Processus de Poisson

Cette page suit uniquement les résultats introduits dans le **Chapitre 2 — Processus de Poisson** du cours de Processus Aléatoires.

## 1. Définition du processus de Poisson

Le chapitre considère un processus de dénombrement $(N(t))_{t\geq0}$. Pour un taux $\lambda>0$, la définition étudiée impose :

1. $N(0)=0$ p.s. ;
2. des accroissements indépendants ;
3. pour $s,t\geq0$ et $n\in\mathbb N$,

$$
P(N(t+s)-N(s)=n)=e^{-\lambda t}\frac{(\lambda t)^n}{n!}.
$$

Le troisième point ne dépend que de la longueur $t$ de l'intervalle : les accroissements sont donc stationnaires. fileciteturn324file4L276-L307

Dans OptiFlowX, ce modèle est représenté par :

```python
from optiflowx.stochastic import PoissonProcess

process = PoissonProcess(rate=2.0)
```

Le paramètre `rate` représente exactement le taux $\lambda$ de la définition du cours.

## 2. Loi du nombre d'occurrences

Sur un intervalle de longueur $t$ :

$$
N(t)\sim\mathcal P(\lambda t),
$$

donc

$$
P(N(t)=n)=e^{-\lambda t}\frac{(\lambda t)^n}{n!}.
$$

```python
process.count_probability(n=4, t=3.0)
```

Pour un intervalle $[s,t]$ :

$$
N(t)-N(s)\sim\mathcal P(\lambda(t-s)).
$$

```python
process.increment_probability(n=2, s=1.0, t=4.0)
```

## 3. Temps inter-arrivées

Le chapitre montre que les temps inter-arrivées sont indépendants et de même loi exponentielle de paramètre $\lambda$.

```python
process.interarrival_samples(10)
```

Si $T_1,T_2,\ldots$ désignent les temps inter-arrivées, les temps d'occurrence sont obtenus par sommes cumulées :

```python
process.arrival_times(10)
```

La simulation du processus utilise directement cette construction :

```python
process.simulate(t_max=10.0)
```

## 4. Conditionnement sur les occurrences

Le chapitre étudie la loi des temps d'arrivée conditionnellement au nombre d'occurrences.

En particulier, conditionnellement à $N(s)=1$, le premier temps d'arrivée est uniforme sur $[0,s]$ :

$$
P(T_1\leq y\mid N(s)=1)=\frac ys,
\qquad 0\leq y\leq s.
$$

```python
process.conditional_first_arrival_cdf(y=1.5, s=3.0)
```

Plus généralement, conditionnellement à $N(s)=k$, les $k$ temps d'arrivée ont la loi des statistiques d'ordre de $k$ variables uniformes indépendantes sur $[0,s]$.

```python
process.conditional_arrival_times(k=4, s=3.0)
```

## 5. Superposition

Le chapitre étudie la superposition de processus de Poisson indépendants. Si les taux sont $\lambda_1$ et $\lambda_2$, le processus superposé a pour taux

$$
\lambda_1+\lambda_2.
$$

```python
p1 = PoissonProcess(2.0)
p2 = PoissonProcess(3.0)
combined = p1.superpose(p2)
```

## 6. Découpage / amincissement

Si chaque occurrence d'un processus de taux $\lambda$ est conservée avec probabilité $p$, le chapitre donne deux processus de taux

$$
p\lambda
\qquad\text{et}\qquad
(1-p)\lambda.
$$

```python
kept, rejected = process.split(0.3)
```

## 7. Processus de Poisson non homogène

Le chapitre introduit ensuite un taux dépendant du temps, $\lambda(t)$, et la fonction moyenne

$$
m(t)=\int_0^t\lambda(x)\,dx.
$$

OptiFlowX représente directement ces deux objets :

```python
from optiflowx.stochastic import NonHomogeneousPoissonProcess

process = NonHomogeneousPoissonProcess(
    intensity=lambda t: 2 * t,
    mean_function=lambda t: t**2,
)

process.mean(3.0)
process.count_probability(2, 3.0)
```

Lorsque $m(t)$ est fourni, le package l'utilise directement. Sinon, il l'approxime numériquement à partir de l'intensité.

## 8. Correspondance cours → package

| Notion du chapitre | Composant OptiFlowX |
|---|---|
| Taux $\lambda$ | `PoissonProcess` |
| $N(t)\sim\mathcal P(\lambda t)$ | `count_probability` |
| Accroissement $N(t)-N(s)$ | `increment_probability` |
| Temps inter-arrivées exponentiels | `interarrival_samples` |
| Temps d'arrivée | `arrival_times` |
| Simulation | `simulate` |
| Arrivée conditionnelle | `conditional_first_arrival_cdf` |
| Statistiques d'ordre conditionnelles | `conditional_arrival_times` |
| Superposition | `superpose` |
| Amincissement | `split` |
| Intensité $\lambda(t)$ | `NonHomogeneousPoissonProcess` |
| Fonction moyenne $m(t)$ | `mean` |
