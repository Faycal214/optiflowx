# Martingales à temps discret

## 1. Filtration

Une filtration est une suite croissante de tribus

$$
\mathcal F_0\subseteq\mathcal F_1\subseteq\cdots.
$$

Dans le cadre fini du package, une filtration est représentée par une suite croissante de partitions.

```python
filtration = Filtration([...])
```

## 2. Processus adapté

Un processus $(X_n)$ est adapté à $(\mathcal F_n)$ si, pour tout $n$, $X_n$ est $\mathcal F_n$-mesurable.

```python
filtration.is_adapted(process)
```

La filtration naturelle est construite à partir de l'information contenue dans le processus jusqu'à l'instant $n$.

```python
Filtration.natural(process)
```

## 3. Martingale, sous-martingale, surmartingale

Le chapitre utilise trois conditions, en plus de l'intégrabilité et de l'adaptation.

Martingale :

$$
E(X_{n+1}\mid\mathcal F_n)=X_n.
$$

Sous-martingale :

$$
E(X_{n+1}\mid\mathcal F_n)\ge X_n.
$$

Surmartingale :

$$
E(X_{n+1}\mid\mathcal F_n)\le X_n.
$$

```python
mart.is_martingale()
mart.is_submartingale()
mart.is_supermartingale()
```

## 4. Caractérisation par les accroissements

La condition de martingale est équivalente à

$$
E[(X_{n+1}-X_n)\mid\mathcal F_n]=0.
$$

OptiFlowX expose le résidu

```python
mart.martingale_residual(n)
```

qui correspond exactement au membre de gauche de cette caractérisation.

## 5. Espérance conditionnelle à plusieurs pas

Le chapitre utilise la propriété de la tour pour obtenir

$$
E[X_{n+k}\mid\mathcal F_n]=X_n
$$

pour une martingale.

```python
mart.conditional_future(n, k)
```

## 6. Martingale de Doob

Si $X$ est intégrable et $(\mathcal F_n)$ est une filtration, alors

$$
X_n=E(X\mid\mathcal F_n)
$$

définit une martingale, appelée martingale de Doob.

```python
Martingale.doob(terminal_variable, filtration)
```

## 7. Transformations convexes et concaves

Le chapitre rappelle l'inégalité de Jensen et montre que, si $(X_n)$ est une martingale et $\varphi$ est convexe avec intégrabilité suffisante, alors $(\varphi(X_n))$ est une sous-martingale.

Pour une fonction concave, on obtient une surmartingale.

```python
transform_martingale(mart, phi)
```

En particulier, les conséquences données dans le cours incluent

$$
(|X_n|)_n
$$

et, lorsqu'il est intégrable,

$$
(X_n^2)_n
$$

comme sous-martingales.

## 8. Monotonie de l'espérance

Pour une surmartingale, $(E[X_n])$ est décroissante ; pour une sous-martingale, elle est croissante.

```python
mart.expectations()
```

## 9. Temps d'arrêt

Un temps d'arrêt $T$ vérifie que

$$
\{T\le n\}\in\mathcal F_n.
$$

Dans la représentation par partitions, cette condition est vérifiée automatiquement lors de la construction.

```python
StoppingTime.from_values(space, values, filtration)
```

Le chapitre montre notamment que, pour deux temps d'arrêt $S,T$, les temps

$$S+T,\qquad S\wedge T,\qquad S\vee T
$$

sont encore des temps d'arrêt.

```python
T.minimum(S)
T.maximum(S)
T.add(S)
```

## 10. Processus arrêté

Le processus arrêté est

$$
X_n^T=X_{n\wedge T}.
$$

```python
stopped = mart.stopped(T)
stopped.values(n)
stopped.sequence()
```

Si $T$ est presque sûrement fini, le chapitre donne la convergence du processus arrêté vers la variable terminale $X_T$ dans le cadre présenté.

```python
stopped.terminal_value()
```

## 11. Composants principaux

| Mathématique | API |
|---|---|
| Filtration | `Filtration` |
| Filtration naturelle | `Filtration.natural` |
| Adaptation | `is_adapted` |
| Martingale | `Martingale` |
| Sous/surmartingale | `is_submartingale`, `is_supermartingale` |
| Résidu conditionnel | `martingale_residual` |
| Espérance future | `conditional_future` |
| Martingale de Doob | `Martingale.doob` |
| Transformation convexe/concave | `transform_martingale` |
| Temps d'arrêt | `StoppingTime` |
| Min/max/somme | `minimum`, `maximum`, `add` |
| Processus arrêté | `StoppedProcess` |
| Variable terminale | `terminal_value` |
