# Chaînes de Markov à temps discret (CMTD)

## 1. Définition

Le chapitre 1 considère un processus stochastique à temps discret et espace d'états discret vérifiant la propriété de Markov :

$$
P(X_{n+1}=j\mid X_0=i_0,\ldots,X_n=i)=P(X_{n+1}=j\mid X_n=i).
$$

Dans le cas homogène, la probabilité de transition ne dépend pas de $n$. On définit la matrice de transition

$$
P=(p_{ij}),\qquad p_{ij}=P(X_{n+1}=j\mid X_n=i).
$$

Chaque ligne de $P$ est une loi de probabilité.

### API

```python
from optiflowx.stochastic import MarkovChain
chain = MarkovChain(P, states=[...])
```

## 2. Loi initiale et transitions à n pas

Si la loi initiale est le vecteur ligne

$$\mu_0=(P(X_0=i))_i,$$

alors

$$
\mu_n=\mu_0P^n.
$$

La matrice $P^n$ contient les probabilités de transition à $n$ pas :

$$
p_{ij}^{(n)}=P(X_{k+n}=j\mid X_k=i).
$$

```python
chain.n_step_transition(n)
chain.state_distribution(mu0, n)
```

## 3. Chapman–Kolmogorov

Pour $m,n\ge 0$ :

$$
P^{(m+n)}=P^{(m)}P^{(n)}.
$$

Dans le cas homogène, cela devient simplement

$$P^{m+n}=P^mP^n.$$

```python
chain.chapman_kolmogorov(m, n)
```

## 4. Accessibilité, communication et classes

On dit que $j$ est accessible depuis $i$ s'il existe $n\ge0$ tel que $p_{ij}^{(n)}>0$.

Deux états communiquent s'ils sont mutuellement accessibles. Les classes de communication sont les ensembles d'états qui communiquent entre eux.

```python
chain.accessible(i, j)
chain.communicate(i, j)
chain.communicating_classes()
```

Une classe fermée est une classe dont aucune transition ne sort de la classe.

```python
chain.closed_classes()
```

## 5. Récurrence et transience

Le chapitre introduit le temps de premier retour $T_i$ après avoir quitté $i$ et distingue les états récurrents et transitoires selon

$$P_i(T_i<\infty)=1
$$

ou une probabilité strictement inférieure à $1$.

```python
chain.classify_states()
```

### Premier retour

Le premier-retour exact à l'instant $n$ est décrit par

$$
f_{ii}^{(n)}=P_i(T_i=n).
$$

```python
first_return_probability(chain, i, n)
```

La probabilité de retour éventuel est

$$
f_i=\sum_{n\ge1}f_{ii}^{(n)}.
$$

```python
return_probability(chain, i)
```

Pour les chaînes finies positives récurrentes, la relation avec la distribution stationnaire est

$$
\mu_i=E_i[T_i]=\frac{1}{\pi_i}.
$$

```python
mean_return_time(chain, i)
```

## 6. Périodicité et ergodicité

La période d'un état est le plus grand commun diviseur des temps de retour possibles.

```python
chain.period(i)
chain.is_aperiodic()
chain.is_ergodic()
```

L'exemple classique du chapitre

$$
P=\begin{pmatrix}0&1\\1&0\end{pmatrix}
$$

possède une distribution stationnaire mais pas de distribution limite, car la chaîne est périodique.

## 7. Distribution stationnaire

Une loi $π$ est stationnaire si

$$
\pi P=\pi,
\qquad \sum_i\pi_i=1,
\qquad \pi_i\ge0.
$$

Pour une chaîne finie irréductible, la loi stationnaire est unique.

```python
chain.stationary_distribution()
```

Pour une chaîne réductible, le chapitre permet plusieurs lois stationnaires. OptiFlowX expose une loi stationnaire pour chaque classe fermée récurrente : toute combinaison convexe de ces lois est encore stationnaire.

```python
stationary_distributions(chain)
```

## 8. Distribution limite

Dans le cas ergodique fini,

$$
\lim_{n\to\infty}p_{ij}^{(n)}=\pi_j.
$$

```python
chain.limiting_distribution()
```

La fonction refuse explicitement les situations où les conditions du chapitre ne permettent pas d'affirmer l'existence d'une limite.

## 9. Probabilités d'absorption

Pour une classe fermée ergodique $C_k$ et un état transitoire $i$, le chapitre définit la probabilité d'absorption par

$$
\pi_i(C_k)=P_i(\exists n\ge1:X_n\in C_k).
$$

```python
chain.absorption_probability(i, C_k)
```

## 10. Fréquences de visite

Pour une trajectoire $X_0,\ldots,X_{N-1}$, la fréquence empirique d'un état $i$ est

$$
\frac1N\sum_{n=0}^{N-1}\mathbf 1_{\{X_n=i\}}.
$$

Dans le cadre ergodique/récurrent positif du chapitre, cette fréquence converge presque sûrement vers $π_i$.

```python
empirical_state_frequencies(path, chain.states)
```

## 11. Composants principaux

| Mathématique | API |
|---|---|
| Matrice $P$ | `MarkovChain` |
| $P^n$ | `n_step_transition` |
| Loi $μ_n$ | `state_distribution` |
| Chapman–Kolmogorov | `chapman_kolmogorov` |
| Accessibilité | `accessible` |
| Communication | `communicate` |
| Classes | `communicating_classes` |
| Récurrence/transience | `classify_states` |
| Période | `period` |
| Stationnaire | `stationary_distribution` |
| Limite | `limiting_distribution` |
| Premier retour | `first_return_probability` |
| Temps moyen de retour | `mean_return_time` |
| Absorption | `absorption_probability` |
| Simulation | `simulate` |
