# Processus de Poisson

## 1. Définition

Le chapitre présente le processus de Poisson comme un processus à temps continu et à valeurs entières positives. Pour un taux $\lambda>0$, le nombre d'événements sur un intervalle de longueur $t$ suit une loi de Poisson de paramètre $\lambda t$ :

$$
P(N(t)=n)=e^{-\lambda t}\frac{(\lambda t)^n}{n!}.
$$

```python
from optiflowx.stochastic import PoissonProcess
process = PoissonProcess(rate=2.0)
process.count_probability(4, 3.0)
```

## 2. Incréments

Pour $0\le s\le t$ :

$$
N(t)-N(s)\sim \mathcal P(\lambda(t-s)).
$$

Les incréments sur des intervalles disjoints sont indépendants dans la définition étudiée.

```python
process.increment_probability(n, s, t)
```

## 3. Temps inter-arrivées

Si $T_1,T_2,\ldots$ sont les temps inter-arrivées, le chapitre montre qu'ils sont indépendants et identiquement distribués selon une loi exponentielle de paramètre $\lambda$.

```python
process.interarrival_samples(k)
process.arrival_times(k)
```

Les temps d'arrivée sont les sommes cumulées des temps inter-arrivées.

## 4. Simulation

Le processus peut être simulé directement par des temps inter-arrivées exponentiels.

```python
process.simulate(t_max)
```

Le résultat contient les instants d'arrivée avant $t_{max}$.

## 5. Conditionnement sur le nombre d'arrivées

Le chapitre établit que, conditionnellement à une arrivée dans $[0,s]$, le premier instant d'arrivée est uniforme sur $[0,s]$.

Ainsi, pour $0\le y\le s$ :

$$
P(T_1\le y\mid N(s)=1)=\frac ys.
$$

```python
process.conditional_first_arrival_cdf(y, s)
```

Plus généralement, conditionnellement à $N(s)=k$, les $k$ instants d'arrivée ont la même loi que les statistiques d'ordre de $k$ variables uniformes indépendantes sur $[0,s]$.

```python
process.conditional_arrival_times(k, s)
```

## 6. Superposition

Si deux processus de Poisson indépendants ont respectivement les taux $\lambda_1$ et $\lambda_2$, leur superposition est un processus de Poisson de taux

$$
\lambda_1+\lambda_2.
$$

```python
combined = process1.superpose(process2)
```

## 7. Découpage / thinning

Si chaque événement d'un processus de taux $\lambda$ est conservé avec probabilité $p$, le processus conservé est de taux

$$p\lambda,
$$

et le processus complémentaire est de taux

$$(1-p)\lambda.$$

```python
first, second = process.split(p)
```

## 8. Processus de Poisson non homogène

Le chapitre introduit l'intensité $\lambda(t)$ et la fonction moyenne

$$
m(t)=\int_0^t\lambda(x)\,dx.
$$

Le nombre d'événements sur $[0,t]$ est alors de loi de Poisson de paramètre $m(t)$ dans le cadre présenté dans le chapitre.

```python
from optiflowx.stochastic import NonHomogeneousPoissonProcess
process = NonHomogeneousPoissonProcess(
    intensity=lambda t: 2*t,
    mean_function=lambda t: t**2,
)
```

La probabilité d'un comptage donné est calculée en remplaçant $\lambda t$ par $m(t)$.

## 9. Composants principaux

| Mathématique | API |
|---|---|
| Taux $\lambda$ | `PoissonProcess` |
| Loi de $N(t)$ | `count_probability` |
| Loi d'un incrément | `increment_probability` |
| Temps inter-arrivée | `interarrival_samples` |
| Temps d'arrivée | `arrival_times` |
| Simulation | `simulate` |
| Arrivées conditionnelles | `conditional_first_arrival_cdf`, `conditional_arrival_times` |
| Superposition | `superpose` |
| Découpage | `split` |
| Intensité $\lambda(t)$ | `NonHomogeneousPoissonProcess` |
| Fonction moyenne $m(t)$ | `mean` |
