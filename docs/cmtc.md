# Chaînes de Markov à temps continu (CMTC)

Cette page suit le **Chapitre 3 — Chaînes de Markov à temps continu**. Les définitions et résultats mathématiques présentés ici sont ceux du PDF du cours ; l'API est ensuite mise en regard de ces objets.

## 1. Processus et matrice de transition

Pour une CMTC homogène, le chapitre utilise les probabilités

$$
p_{ij}(t)=P(X_{s+t}=j\mid X_s=i),
$$

qui ne dépendent que de la durée $t$. On note

$$
P(t)=(p_{ij}(t)).
$$

La matrice vérifie notamment $P(0)=I$ et la propriété de Chapman–Kolmogorov.

```python
from optiflowx.stochastic import ContinuousTimeMarkovChain

chain = ContinuousTimeMarkovChain(Q, states=[0, 1, 2])
chain.transition_matrix(2.0)
```

## 2. Générateur infinitésimal

Le chapitre introduit la matrice génératrice $Q=(q_{ij})$ par le comportement infinitésimal. Pour $i\neq j$ :

$$
p_{ij}(h)=q_{ij}h+o(h).
$$

Les coefficients hors diagonale sont positifs ou nuls, les coefficients diagonaux sont non positifs, et les lignes de $Q$ somment à zéro.

```python
chain.generator_matrix
chain.infinitesimal_transition_matrix(h)
```

L'expression `I + hQ` représente ici le développement au premier ordre utilisé dans le cours ; la matrice de transition exacte est traitée séparément.

## 3. Matrice de transition et loi à l'instant t

Dans le cadre homogène fini du chapitre :

$$
P(t)=e^{tQ}.
$$

```python
P_t = chain.transition_matrix(t)
```

Si la loi initiale est le vecteur-ligne $\mu_0$, alors

$$
\mu_t=\mu_0P(t).
$$

```python
mu_t = chain.state_distribution(mu_0, t)
```

## 4. Chapman–Kolmogorov

L'homogénéité donne

$$
P(s+t)=P(s)P(t).
$$

```python
chain.chapman_kolmogorov(s, t)
```

## 5. Équations de Kolmogorov

Le chapitre présente les équations avant et arrière :

$$
\frac{dP(t)}{dt}=P(t)Q,
$$

et

$$
\frac{dP(t)}{dt}=QP(t).
$$

```python
chain.forward_derivative(t)
chain.backward_derivative(t)
```

## 6. Temps de séjour

Depuis l'état $i$, le taux de sortie est

$$
q_i=-q_{ii}.
$$

Le temps de séjour est exponentiel de paramètre $q_i$ lorsque $q_i>0$.

```python
chain.holding_rate(i)
chain.holding_time(i)
```

Si le taux de sortie est nul, le temps de séjour est infini dans la construction utilisée par le package.

## 7. Chaîne des sauts

Le chapitre associe à la CMTC une chaîne de Markov discrète décrivant les états visités aux instants de saut. Pour $i\neq j$ et $q_i>0$ :

$$
\widetilde p_{ij}=\frac{q_{ij}}{q_i}
=\frac{q_{ij}}{-q_{ii}}.
$$

```python
chain.jump_chain_matrix()
jump_chain = chain.jump_chain()
```

Cette relation est importante car les propriétés de communication et de récurrence étudiées dans le chapitre sont reliées à la chaîne des sauts.

## 8. Loi stationnaire

Une loi $\pi$ est stationnaire si

$$
\pi P(t)=\pi,\qquad \forall t\geq0.
$$

Dans le cadre étudié, elle est caractérisée par

$$
\pi Q=0,
\qquad \sum_i\pi_i=1.
$$

```python
pi = chain.stationary_distribution()
```

Le package résout cette relation dans le cadre fini représenté par `ContinuousTimeMarkovChain`.

## 9. Relation avec la chaîne des sauts

Le chapitre relie la distribution stationnaire de la CMTC à celle de sa chaîne des sauts en tenant compte des temps de séjour. OptiFlowX conserve cette relation dans les outils de théorie associés à la CMTC.

## 10. Explosion et régularité

Le chapitre définit le temps d'explosion à partir de la somme des temps de séjour successifs :

$$
\zeta=\sum_{n\geq1}S_n.
$$

Le processus est régulier lorsque

$$
P(\zeta=+\infty)=1,
$$

et explosif lorsqu'une explosion en temps fini peut avoir une probabilité positive.

Pour les processus de naissance pure, la question est liée à la série des inverses des taux de naissance. Le package ne déduit pas la convergence d'une série infinie à partir d'un nombre fini de termes : il expose explicitement une somme partielle dans `pure_birth_reciprocal_rate_sum`.

## 11. Comportement ergodique dans le temps

Pour un état $i$, le temps passé dans $i$ jusqu'à $T$ peut être représenté par

$$
\int_0^T \mathbf 1_{\{X_t=i\}}\,dt.
$$

La fraction de temps est

$$
\frac1T\int_0^T \mathbf 1_{\{X_t=i\}}\,dt.
$$

Dans le cadre ergodique/récurrent positif développé dans le chapitre, cette quantité converge vers la probabilité stationnaire correspondante.

## 12. Simulation

La simulation suit la construction du chapitre :

1. choisir l'état courant ;
2. générer le temps de séjour exponentiel associé à son taux de sortie ;
3. choisir le prochain état selon la chaîne des sauts ;
4. recommencer jusqu'à l'horizon demandé.

```python
path = chain.simulate(t_max=20.0, initial_state=0)
path.times
path.states
path.state_at(5.0)
```

## 13. Correspondance cours → package

| Objet du chapitre | Composant OptiFlowX |
|---|---|
| Générateur $Q$ | `ContinuousTimeMarkovChain` |
| Approximation infinitésimale | `infinitesimal_transition_matrix` |
| $P(t)$ | `transition_matrix` |
| $p_{ij}(t)$ | `transition_probability` |
| Loi $\mu_t$ | `state_distribution` |
| Chapman–Kolmogorov | `chapman_kolmogorov` |
| Kolmogorov avant | `forward_derivative` |
| Kolmogorov arrière | `backward_derivative` |
| Temps de séjour | `holding_rate`, `holding_time` |
| Chaîne des sauts | `jump_chain`, `jump_chain_matrix` |
| Loi stationnaire | `stationary_distribution` |
| Trajectoire | `CTMCPath` |
| Simulation | `simulate` |
