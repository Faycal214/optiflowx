# Martingales à temps discret

Cette page suit uniquement le **Chapitre 5 — Martingales à temps discret**. Le niveau d'abstraction est conservé : filtration, adaptation, martingale, sous-martingale/surmartingale, propriétés conditionnelles, temps d'arrêt et processus arrêté.

## 1. Filtration

Le cours définit une filtration comme une suite croissante de sous-tribus :

$$
\mathcal F_0\subseteq\mathcal F_1\subseteq\cdots.
$$

fileciteturn347file4L321-L337

Dans OptiFlowX, le cadre fini représente chaque $\mathcal F_n$ par une `Partition` :

```python
filtration = Filtration([F0, F1, F2])
```

La condition d'inclusion des tribus est représentée par la relation de raffinement des partitions.

## 2. Processus adapté

Le chapitre dit que $(X_n)$ est adapté à $(\mathcal F_n)$ si, pour tout $n$,

$$
X_n\text{ est }\mathcal F_n\text{-mesurable}.
$$

fileciteturn347file4L327-L337

```python
filtration.is_adapted(process)
```

La filtration naturelle est construite à partir de l'information observée par le processus jusqu'au temps $n$ :

```python
Filtration.natural(process)
```

## 3. Définition d'une martingale

Dans le chapitre, une martingale doit être intégrable, adaptée et satisfaire

$$
E[X_{n+1}\mid\mathcal F_n]=X_n.
$$

Une formulation équivalente utilisée dans le cours est

$$
E[(X_{n+1}-X_n)\mid\mathcal F_n]=0.
$$

fileciteturn347file3L246-L262

OptiFlowX vérifie directement le résidu :

```python
mart.martingale_residual(n)
mart.is_martingale()
```

## 4. Sous-martingale et surmartingale

Le chapitre définit une surmartingale par

$$
E[X_{n+1}\mid\mathcal F_n]\leq X_n
$$

et une sous-martingale par

$$
E[X_{n+1}\mid\mathcal F_n]\geq X_n,
$$

avec les conditions d'intégrabilité et d'adaptation. fileciteturn324file2L145-L154

```python
mart.is_submartingale()
mart.is_supermartingale()
```

## 5. Espérance conditionnelle à plusieurs pas

Pour une martingale, le chapitre établit

$$
E[X_{n+k}\mid\mathcal F_n]=X_n,
\qquad k\geq0.
$$

fileciteturn324file1L76-L97

```python
mart.conditional_future(n, k)
```

## 6. Martingale de Doob

Si $(\mathcal F_n)$ est une filtration et $X$ une variable aléatoire intégrable, le chapitre donne l'exemple

$$
X_n=E(X\mid\mathcal F_n),
$$

qui définit une martingale appelée martingale de Doob. fileciteturn347file2L196-L217

```python
mart = Martingale.doob(terminal_variable, filtration)
```

## 7. Transformations par fonctions convexes ou concaves

Le cours rappelle l'inégalité de Jensen et établit que, sous les hypothèses d'intégrabilité du chapitre, une transformation convexe d'une martingale donne une sous-martingale ; une transformation concave donne une surmartingale. fileciteturn324file9L605-L619

Cette propriété est documentée ici comme un **résultat théorique du cours**. Elle ne correspond pas encore à une fonction publique dédiée dans l'API actuelle ; nous ne prétendons donc pas qu'une méthode de transformation existe.

## 8. Exemple du cours : marche aléatoire

Le chapitre considère une marche symétrique

$$
X_n=\sum_{i=1}^{n}\xi_i,
$$

avec des variables $\xi_i$ IID prenant les valeurs $-1$ et $1$ avec probabilité $1/2$, et sa filtration canonique. fileciteturn324file1L99-L127

Le package peut représenter cet exemple dans le cadre fini et tester la condition de martingale par `Martingale.is_martingale()`.

## 9. Temps d'arrêt

Un temps d'arrêt $T$ est étudié relativement à une filtration. OptiFlowX représente ses valeurs sur les issues de l'espace fini :

```python
T = StoppingTime.from_values(
    space,
    values,
    filtration,
)
```

La construction vérifie que les événements correspondant aux temps considérés sont compatibles avec l'information disponible à ces temps.

Le package expose également les opérations étudiées dans le chapitre :

```python
T.minimum(S)   # T ∧ S
T.maximum(S)   # T ∨ S
T.add(S)       # T + S
```

## 10. Processus arrêté

Pour un processus adapté $(X_n)$ et un temps d'arrêt $T$, le chapitre définit

$$
X_n^T=X_{n\wedge T}.
$$

De manière équivalente,

$$
X_n^T
=
X_n\mathbf1_{\{n<T\}}
+X_T\mathbf1_{\{T\leq n\}}.
$$

fileciteturn347file0L33-L67

```python
stopped = mart.stopped(T)
stopped.values(n)
stopped.sequence()
```

Le théorème du chapitre établit que si $(X_n)$ est une martingale et $T$ un temps d'arrêt, alors le processus arrêté reste une martingale. fileciteturn347file0L69-L75

## 11. Variable terminale

Lorsque $T$ est presque sûrement fini, le chapitre définit la variable terminale $X_T$ et établit la convergence du processus arrêté vers cette variable terminale. fileciteturn324file8L536-L577

```python
terminal = stopped.terminal_value()
```

Si une issue donne $T=+\infty$, `terminal_value()` refuse de fabriquer une valeur terminale qui n'est pas définie dans ce cadre.

## 12. Correspondance cours → package

| Objet du chapitre | Composant OptiFlowX |
|---|---|
| Filtration | `Filtration` |
| Filtration naturelle | `Filtration.natural` |
| Processus adapté | `is_adapted` |
| Martingale | `Martingale` |
| Condition martingale | `conditional_next`, `martingale_residual` |
| Sous-martingale | `is_submartingale` |
| Surmartingale | `is_supermartingale` |
| Espérance future | `conditional_future` |
| Martingale de Doob | `Martingale.doob` |
| Temps d'arrêt | `StoppingTime` |
| $T\wedge S$ | `minimum` |
| $T\vee S$ | `maximum` |
| $T+S$ | `add` |
| Processus arrêté | `StoppedProcess` |
| $X^T_n$ | `values`, `sequence` |
| Variable terminale $X_T$ | `terminal_value` |
