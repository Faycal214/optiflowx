<details>
<summary>Course slide 12</summary>

<pre>
Propriété

Proposition
Soit 0 = τ 0 &lt; τ 1 &lt;        &lt; τ n &lt; . . . les instants d’occurrence d’un processus de Poisson de taux λ. Alors, la suite des inter-occurrences est une suite de v .a. indépendantes et de loi commune exponentielle de paramètre λ.

N. Boussaha () Processus Aléatoires (2) USTHB, 2024-2025 12 / 22
</pre>
</details>
<details>
<summary>Course slide 13</summary>
<pre>
Démonstration.
Soient i 6= j. On considère P(Ti &gt; x, Tj &gt; y). Par indépendance des accroissements du processus de Poisson :
P(Ti &gt; x, Tj &gt; y ) = P(N(x)=0)P(N(y)=0).
Comme N(t) suit une loi P(λt), on obtient P(Ti &gt; x, Tj &gt; y)=e^{-λx}e^{-λy}=P(Ti &gt; x)P(Tj &gt; y).
Cela prouve que les inter-occurrences sont indépendantes et exponentielles de paramètre λ.

N. Boussaha () Processus Aléatoires (2) USTHB, 2024-2025 13 / 22
</pre>
</details>
<details>
<summary>Course slide 14</summary>
<pre>
Propriété

Proposition
Soit (N(t))t≥0 un PP de taux λ. La loi conditionnelle de l’instant de la 1ère occurrence sachant qu’une seule occurrence a eu lieu jusqu’à l’instant s, est la loi uniforme sur [0,s].

N. Boussaha () Processus Aléatoires (2) USTHB, 2024-2025 14 / 22
</pre>
</details>
<details>
<summary>Course slide 15</summary>
<pre>
Propriété
Démonstration.
Soit Y l’instant de la 1ère occurrence. Pour 0&lt;y&lt;s, par définition de Y et par la formule de probabilité conditionnelle, on obtient :
P(Y≤y | N(s)=1)=P(N(y)=1)P(N(s-y)=0)/P(N(s)=1)=y/s.

N. Boussaha () Processus Aléatoires (2) USTHB, 2024-2025 15 / 22
</pre>
</details>
<details>
<summary>Course slide 16</summary>
<pre>
Propriété
Démonstration.
Si y≤0, P(Y≤y | N(s)=1)=0. Si y≥s, P(Y≤y | N(s)=1)=1. Ce qui montre que Y suit une loi uniforme sur [0,s].

N. Boussaha () Processus Aléatoires (2) USTHB, 2024-2025 16 / 22
</pre>
</details>
