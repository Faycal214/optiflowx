# Chapitre 1 — Chaînes de Markov à temps discret (CMTD)

> **Source de référence:** *Processus Aléatoires (1), USTHB, 2024–2025, N. Boussaha.*
> Cette page présente d'abord les notions telles qu'elles sont développées dans le PDF, puis montre comment OptiFlowX les implémente.

## 1. Objet du chapitre

Le chapitre étudie les chaînes de Markov à temps discret et espace d'états discret. Pour une chaîne homogène, les probabilités de transition ne dépendent pas du temps.

La propriété de Markov est l'idée centrale : conditionnellement à l'état présent, le futur ne dépend pas du passé.

Dans le cadre discret, la chaîne est décrite par des **probabilités de transition** et une **matrice de transition**. Le chapitre introduit ensuite les transitions à plusieurs étapes, les équations de Chapman–Kolmogorov, l'accessibilité et la communication, la récurrence/transience, la périodicité, les distributions stationnaires, les distributions limites et l'absorption. [Chapitre 1, sections correspondantes du PDF.]

---

## 2. Matrice de transition

Pour une chaîne homogène sur un espace d'états `S`, on note

$$
p_{ij}=P(X_{n+1}=j\mid X_n=i).
$$

La matrice de transition est

$$
P=(p_{ij})_{i,j\in S}.
$$

Chaque ligne est une loi de probabilité : les coefficients sont non négatifs et leur somme vaut `1`.

### Dans OptiFlowX

```python
from optiflowx.stochastic import MarkovChain

chain = MarkovChain(
    [[0.7, 0.3],
     [0.4, 0.6]],
    states=["A", "B"],
)
```

La classe `MarkovChain` représente donc exactement l'objet matriciel utilisé dans le chapitre.

---

## 3. Transitions en plusieurs étapes

Le PDF définit

$$
p_{ij}^{(n)}=P(X_{n+m}=j\mid X_m=i),
$$

et la matrice

$$
P^{(n)}=(p_{ij}^{(n)})_{i,j\in S}.
$$

Dans le cas homogène, les équations de Chapman–Kolmogorov donnent

$$
P^{(m+n)}=P^{(m)}P^{(n)},
$$

et donc

$$
P^{(n)}=P^n.
$$

Le PDF insiste sur cette distinction de notation : $p_{ij}^{(n)}$ désigne une probabilité à `n` étapes, tandis que $(P^n)_{ij}$ est le calcul matriciel correspondant.

### Dans OptiFlowX

```python
P5 = chain.n_step_transition(5)
```

La méthode calcule directement `P^5`.

---

## 4. Loi de la chaîne

Si

$$
\mu_n=(P(X_n=i))_{i\in S}
$$

est la loi de $X_n$, alors le chapitre obtient

$$
\mu_n=\mu_{n-1}P
$$

et donc

$$
\boxed{\mu_n=\mu_0P^n}.
$$

### Dans OptiFlowX

```python
mu_n = chain.state_distribution(mu_0, n=10)
```

Cette méthode correspond directement à la relation de récurrence du cours.

---

## 5. Accessibilité et classes de communication

Le chapitre définit l'accessibilité à partir des probabilités de transition à plusieurs étapes : l'état `j` est accessible depuis `i` lorsqu'il existe un entier `n` pour lequel

$$
p_{ij}^{(n)}>0.
$$

Deux états communiquent lorsque chacun est accessible depuis l'autre. Les classes de communication regroupent les états qui communiquent entre eux.

Une classe de communication est dite **fermée** lorsque la chaîne ne peut pas en sortir.

### Dans OptiFlowX

```python
chain.accessible("A", "B")
chain.communicate("A", "B")
chain.communicating_classes()
chain.closed_classes()
```

La recherche des classes est une représentation informatique de la structure de communication introduite dans le PDF.

---

## 6. Récurrence et transience

Le chapitre définit la probabilité de retour et distingue les états récurrents et transitoires. Un état `j` est récurrent si, en partant de `j`, on revient à `j` presque sûrement ; sinon il est transitoire.

Le PDF donne également le critère

$$
\sum_{n=1}^{\infty}p_{jj}^{(n)}=
\begin{cases}
\infty, & \text{si }j\text{ est récurrent},\\
<\infty, & \text{si }j\text{ est transitoire}.
\end{cases}
$$

Il distingue ensuite :

- récurrence positive : le temps moyen de retour est fini ;
- récurrence nulle : le temps moyen de retour est infini.

Pour une chaîne finie, OptiFlowX utilise la structure des classes communicantes pour obtenir cette classification.

### Dans OptiFlowX

```python
chain.classify_states()
```

Le résultat donne, pour chaque état, `"recurrent"` ou `"transient"`.

---

## 7. Premier retour et temps moyen de retour

Le temps de premier retour permet de distinguer les retours exacts à différents instants. OptiFlowX expose

```python
chain.first_visit_probability(i, j, n)
```

pour le premier passage vers `j` à l'instant `n`.

La probabilité de retour éventuel est calculée en sommant les probabilités de premier retour.

```python
from optiflowx.stochastic import return_probability

return_probability(chain, "A")
```

Dans le cas irréductible récurrent positif, le PDF établit la relation

$$
\boxed{\pi_j=\frac{1}{\mu_j}},
$$

où `\mu_j` est le temps moyen de retour à `j`.

Le chapitre interprète alors `\pi_j` comme la proportion de temps passée dans l'état `j` à long terme. Cette relation est explicitement donnée dans la partie « Distribution stationnaire » du PDF. fileciteturn308file3L351-L405

---

## 8. Périodicité

La période d'un état `i` est définie dans le PDF par

$$
d(i)=\operatorname{pgcd}\{n\ge1:p_{ii}^{(n)}>0\}.
$$

Un état est apériodique si sa période vaut `1`. L'apériodicité est une propriété de classe lorsque des états communiquent.

### Dans OptiFlowX

```python
chain.period("A")
chain.is_aperiodic()
```

---

## 9. Distribution stationnaire

C'est un point important : **la documentation distingue explicitement la définition mathématique du mécanisme de calcul Python.**

### Définition du cours

Une distribution `\pi` est stationnaire lorsqu'elle reste inchangée par la transition :

$$
\boxed{\pi P=\pi}.
$$

Dans le chapitre, pour une chaîne irréductible et récurrente positive, la distribution stationnaire est unique et vérifie

$$
\pi_j=\frac{1}{\mu_j}.
$$

Le PDF donne aussi le cas d'une classe de communication fermée finie : il existe alors une distribution stationnaire unique concentrée sur cette classe. fileciteturn308file3L351-L405

### Ce que fait la classe

```python
pi = chain.stationary_distribution()
```

Cette méthode résout numériquement le système

$$
\pi P=\pi,
\qquad
\sum_i\pi_i=1.
$$

Elle est volontairement limitée au cas irréductible dans la classe `MarkovChain`, car c'est le cas où le cours établit l'unicité de la distribution stationnaire.

Pour une chaîne réductible, OptiFlowX fournit séparément les distributions associées aux classes fermées via la fonction `stationary_distributions`.

> **Important:** le code ne définit pas ce qu'est une distribution stationnaire. Il calcule l'objet défini dans le cours.

---

## 10. Distribution limite

Le chapitre demande sous quelles conditions la limite de la loi existe.

Pour une chaîne ergodique, le PDF énonce

$$
\lim_{n\to\infty}P^n=P,
$$

où la matrice limite possède des lignes identiques et leur valeur commune est la distribution stationnaire :

$$
\lim_{n\to\infty}p_{ij}^{(n)}=\pi_j.
$$

Le PDF généralise également ce résultat à une chaîne possédant une seule classe fermée ergodique, tous les autres états étant transitoires. fileciteturn307file0L40-L120

Le cours montre aussi qu'une distribution stationnaire peut exister sans que la distribution limite existe : la chaîne alternante

$$
P=\begin{pmatrix}0&1\\1&0\end{pmatrix}
$$

possède une distribution stationnaire mais reste périodique. fileciteturn308file6L678-L706

### Dans OptiFlowX

```python
limit = chain.limiting_distribution()
```

La méthode lève une erreur lorsque les conditions prises en charge par le chapitre ne permettent pas d'affirmer une limite.

---

## 11. Absorption

Le PDF considère, notamment, un état ou une classe fermée vers laquelle une chaîne peut être absorbée. Pour une classe fermée ergodique `C_k` et un état transitoire `i`, il définit une probabilité d'absorption obtenue à partir des probabilités d'atteindre cette classe. fileciteturn308file6L721-L744

### Dans OptiFlowX

```python
prob = chain.absorption_probability(i, C_k)
```

Le calcul numérique exploite le système linéaire associé à la chaîne finie. La signification reste celle du cours : probabilité de finir dans la classe considérée.

---

## 12. Fréquences de visite

Dans la partie asymptotique, le cours relie la distribution stationnaire au comportement de long terme et à la proportion de temps passée dans les états. fileciteturn308file3L386-L402

Pour une trajectoire simulée `X_0, ..., X_{N-1}`, on peut estimer la fréquence d'un état `i` par

$$
\widehat\pi_i=
\frac{1}{N}
\sum_{n=0}^{N-1}\mathbf 1_{\{X_n=i\}}.
$$

```python
from optiflowx.stochastic import empirical_state_frequencies

frequencies = empirical_state_frequencies(path, chain.states)
```

Cette quantité est une **estimation empirique** : elle ne remplace pas la définition de `\pi` et ne constitue pas, à elle seule, une preuve de convergence.

---

## 13. Exemple guidé : distribution stationnaire

Considérons la chaîne

$$
P=
\begin{pmatrix}
0 & 1/2 & 1/2\\
1/2 & 0 & 1/2\\
1/2 & 1/2 & 0
\end{pmatrix}.
$$

Le PDF donne comme distribution stationnaire

$$
\pi=\left(\frac13,\frac13,\frac13\right).
$$

L'interprétation donnée est que, sur le long terme, la proportion de temps passée dans chaque état est la même. fileciteturn308file3L403-L415

Dans OptiFlowX :

```python
chain = MarkovChain(
    [
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ],
    states=[1, 2, 3],
)

print(chain.stationary_distribution())
```

L'important est la correspondance :

$$
\boxed{\text{définition du PDF}}
\quad\longleftrightarrow\quad
\boxed{\text{objet Python}}
\quad\longleftrightarrow\quad
\boxed{\text{calcul numérique}}.
$$

---

## 14. Carte mathématique → API

| Notion du cours | OptiFlowX |
|---|---|
| Matrice de transition `P` | `MarkovChain` |
| Transition à `n` étapes `P^(n)` | `n_step_transition` |
| Loi `\mu_n=\mu_0P^n` | `state_distribution` |
| Chapman–Kolmogorov | `chapman_kolmogorov` |
| Accessibilité | `accessible` |
| Communication | `communicate` |
| Classes | `communicating_classes` |
| Classes fermées | `closed_classes` |
| Récurrence / transience | `classify_states` |
| Période | `period` |
| Distribution stationnaire | `stationary_distribution` |
| Distribution limite | `limiting_distribution` |
| Probabilité d'absorption | `absorption_probability` |
| Simulation | `simulate` |

---

## 15. Règle documentaire OptiFlowX

Pour chaque nouvelle notion du package, la documentation doit suivre cet ordre :

1. **Définition du PDF** — sans ajouter une nouvelle définition externe.
2. **Notation et formule** — même objet mathématique que dans le cours.
3. **Hypothèses** — préciser dans quel cadre le résultat du cours s'applique.
4. **Implémentation** — expliquer quelle classe/méthode représente l'objet.
5. **Exemple** — montrer un cas numérique.
6. **Limites** — préciser quand l'API refuse de conclure.

Cette structure permet de faire d'OptiFlowX un support de cours exécutable plutôt qu'une simple documentation de bibliothèque.
