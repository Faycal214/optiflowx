# Chaînes de Markov à temps continu (CMTC)

## 1. Définition

Une CMTC homogène est un processus $(X_t)_{t\ge0}$ à espace d'états discret vérifiant la propriété de Markov. Dans le cas homogène,

$$
P(X_{t+s}=j\mid X_s=i)=p_{ij}(t),
$$

et on note

$$P(t)=(p_{ij}(t)).
$$

La matrice satisfait $P(0)=I$ et les probabilités de chaque ligne somment à $1$.

```python
from optiflowx.stochastic import ContinuousTimeMarkovChain
chain = ContinuousTimeMarkovChain(Q, states=[...])
```

## 2. Générateur infinitésimal

Le générateur $Q=(q_{ij})$ vérifie, pour $i\ne j$,

$$
p_{ij}(h)=q_{ij}h+o(h),
$$

et les lignes de $Q$ somment à zéro. Les coefficients diagonaux sont non positifs et les coefficients hors diagonale sont non négatifs.

```python
chain.generator_matrix
chain.infinitesimal_transition_matrix(h)
```

## 3. Matrice de transition à temps t

Dans le cadre fini homogène étudié :

$$
P(t)=e^{Qt}.
$$

```python
chain.transition_matrix(t)
```

La loi initiale $\mu_0$ donne

$$
\mu_t=\mu_0P(t).
$$

```python
chain.state_distribution(mu0, t)
```

## 4. Chapman–Kolmogorov

Pour $s,t\ge0$ :

$$
P(s+t)=P(s)P(t).
$$

```python
chain.chapman_kolmogorov(s, t)
```

## 5. Équations de Kolmogorov

Le chapitre introduit les équations avant et arrière :

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

Depuis un état $i$, le temps de séjour avant le prochain saut est exponentiel de paramètre

$$
q_i=-q_{ii}.
$$

```python
chain.holding_rate(i)
chain.holding_time(i)
```

Un état avec taux de sortie nul est absorbant dans la représentation finie utilisée ici et son temps de séjour est infini.

## 7. Chaîne des sauts

Conditionnellement au fait qu'un saut quitte $i$, la probabilité que le prochain état soit $j$ est

$$
\tilde p_{ij}=\frac{q_{ij}}{-q_{ii}},\qquad i\ne j.
$$

```python
chain.jump_chain_matrix()
chain.jump_chain()
```

Le chapitre établit également que la communication du processus est équivalente à celle de sa chaîne des sauts.

## 8. Communication et récurrence

Les notions d'accessibilité, de communication et de classe s'étendent au temps continu. Le chapitre relie ces propriétés au graphe de la chaîne des sauts.

```python
ctmc_communication_classes(chain)
```

Le temps de premier retour est désormais une variable aléatoire continue. La récurrence signifie un retour presque sûr après avoir quitté l'état.

## 9. Distribution stationnaire

Une distribution $\pi$ est stationnaire si

$$
\pi P(t)=\pi,\qquad \forall t\ge0.
$$

Dans le cadre fini étudié, ceci est équivalent à

$$
\pi Q=0,
\qquad \sum_i\pi_i=1.
$$

```python
chain.stationary_distribution()
```

## 10. Relation avec la chaîne des sauts

Si la chaîne des sauts possède une loi stationnaire $\phi$ et si $q_i=-q_{ii}>0$, le chapitre donne

$$
\pi_i=\frac{\phi_i/q_i}{\sum_j\phi_j/q_j}.
$$

```python
ctmc_stationary_from_jump_chain(chain)
```

Le temps moyen de retour vérifie

$$
\mu_i=\frac{1}{q_i\pi_i}
$$

avec $q_i=-q_{ii}$.

```python
ctmc_mean_return_time(chain, i)
```

## 11. Explosion

Le chapitre définit le premier temps d'explosion par la somme des temps de séjour successifs :

$$
\zeta=\sum_{n\ge1}S_n.
$$

Le processus est régulier si

$$P(\zeta=+\infty)=1,
$$

et explosif si

$$P(\zeta<+\infty)>0.
$$

OptiFlowX ne transforme pas une troncature numérique en affirmation de convergence : pour les processus de naissance pure, la fonction disponible retourne explicitement une somme partielle des $1/\lambda_n$.

## 12. Ergodicité continue

Pour un état $i$, la fraction de temps passée dans $i$ est

$$
\frac1T\int_0^T\mathbf 1_{\{X_t=i\}}dt.
$$

Dans le cas ergodique/récurrent positif du chapitre,

$$
\frac1T\int_0^T\mathbf 1_{\{X_t=i\}}dt\xrightarrow[T\to\infty]{p.s.}\pi_i.
$$

```python
occupation_fraction(path, i, T)
```

## 13. Composants principaux

| Mathématique | API |
|---|---|
| Générateur $Q$ | `ContinuousTimeMarkovChain` |
| $I+hQ$ | `infinitesimal_transition_matrix` |
| $P(t)$ | `transition_matrix` |
| Loi à t | `state_distribution` |
| Kolmogorov avant | `forward_derivative` |
| Kolmogorov arrière | `backward_derivative` |
| Temps de séjour | `holding_time` |
| Chaîne des sauts | `jump_chain` |
| Stationnaire | `stationary_distribution` |
| Relation chaîne des sauts | `ctmc_stationary_from_jump_chain` |
| Retour moyen | `ctmc_mean_return_time` |
| Occupation | `occupation_time`, `occupation_fraction` |
| Simulation | `simulate` |
