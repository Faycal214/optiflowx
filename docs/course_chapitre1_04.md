<details>
<summary>PDF page 55</summary>

<pre>
Caractérisation des états

Theorem
    L’état i est transitoire si :

                                    ∑ pii
                                          (n )
                                                 &lt; +∞.
                                    n 1



     N. Boussaha ()            Processus Aléatoires (1)   USTHB, 2024-2025   31 / 66
</pre>

</details>
<details>
<summary>PDF page 56</summary>

<pre>
Caractérisation des états

Theorem
    L’état i est transitoire si :

                                      ∑ pii
                                              (n )
                                                     &lt; +∞.
                                     n 1

    L’état i est récurrent nul si :

                      ∑ pii
                            (n )                               (n )
                                   = +∞ et                lim p = 0.
                                                        n !+∞ ii
                      n 1



     N. Boussaha ()                Processus Aléatoires (1)       USTHB, 2024-2025   31 / 66
</pre>

</details>
<details>
<summary>PDF page 57</summary>

<pre>
Caractérisation des états

Theorem
    L’état i est transitoire si :

                                      ∑ pii
                                              (n )
                                                     &lt; +∞.
                                     n 1

    L’état i est récurrent nul si :

                      ∑ pii
                            (n )                               (n )
                                   = +∞ et                lim p = 0.
                                                        n !+∞ ii
                      n 1

    L’état i est récurrent positif si :

                      ∑ pii
                            (n )                               (n )
                                   = +∞ et                lim p &gt; 0.
                                                        n !+∞ ii
                      n 1



     N. Boussaha ()                Processus Aléatoires (1)       USTHB, 2024-2025   31 / 66
</pre>

</details>
<details>
<summary>PDF page 58</summary>

<pre>
Propriété de périodicité
Il s’agit d’étudier dans quelles conditions le temps qui sépare deux retours
au même état i est ou n’est pas multiple d’un temps minimum.
Definition
                                                                         (n )
On définit la période d (i ) d’un état i comme étant le pgcd fn    1 : pii       &gt; 0g.
          (n )
     Si pii = 0 pour tout n 1 alors d (i ) = ∞.
     Si d (i ) = 1 l’état i est apériodique.
     Une chaîne est dite apériodique si tous ses états sont apériodiques.

Proposition
Si i $ j alors d (i ) = d (j ). On dit que la périodicité est une propriété de
classe.

Démonstration.
(TD). Indication : Penser à utiliser les équations de Chapman-Kolmogorov.
      N. Boussaha ()            Processus Aléatoires (1)     USTHB, 2024-2025     32 / 66
</pre>

</details>
<details>
<summary>PDF page 59</summary>

<pre>
Ergodicité



Definition
Un état est dit ergodique s’il est récurrent positif et apériodique.

    Une Chaîne de Markov est dite ergodique si tous ses états sont
    ergodiques.



      N. Boussaha ()         Processus Aléatoires (1)   USTHB, 2024-2025   33 / 66
</pre>

</details>
<details>
<summary>PDF page 60</summary>

<pre>
Chaine de Markov a espace d etats fini



Proposition
Soit |S| &lt; ∞, alors toute chaîne de Markov
  a au moins un état récurrent.
  irréductible est récurrente positive.
  ne peut pas avoir d’états récurrent nul.
  irréductible et apériodique, elle est ergodique.




      N. Boussaha ()           Processus Aléatoires (1)   USTHB, 2024-2025   34 / 66
</pre>

</details>
<details>
<summary>PDF page 61</summary>

<pre>
Distribution stationnaire
Definition
Une distribution π est dite stationnaire si :
                           8
                           &gt;
                           &lt;π = πP,
                              πj    0, 8 j 2 S,
                           &gt;
                           :
                              ∑j 2Sπj = 1.

avec π = (π 1 , π 2 , . . . ), ou d’une façon explicite :
                                 8
                                 &lt;π j = ∑i 2S π i pij ,
                                 &gt;
                                   πj    0, 8 j 2 S,
                                 &gt;
                                 :
                                   ∑j 2S π j = 1.

Conséquence
si π est une distribution stationnaire donc π = πPn , 8n         1.
       N. Boussaha ()             Processus Aléatoires (1)             USTHB, 2024-2025   35 / 66
</pre>

</details>
<details>
<summary>PDF page 62</summary>

<pre>
Distribution stationnaire


Example (Markov)
Markov lui-même a analysé la succession de 20 000 lettres dans le poème
Eugène Onéguine d’A. S. Pouchkine, découvrant que la probabilité
stationnaire d’une voyelle est p = 0, 432, que la probabilité qu’une voyelle
soit suivie d’une voyelle est p1 = 0, 128, et que la probabilité qu’une
voyelle soit suivie d’une consonne est p2 = 0, 663.


Remarque
pour cet exemple :
  Il existe une seule distribution stationnaire.
  La chaîne est récurrente positive et apériodique.



      N. Boussaha ()                Processus Aléatoires (1)   USTHB, 2024-2025   36 / 66
</pre>

</details>
<details>
<summary>PDF page 63</summary>

<pre>
Distribution stationnaire



Example
        0 1
P=                     .
        1 0

Remarque
pour cet exemple :
  Il existe une seule distribution stationnaire.
  La chaîne est récurrente positive et périodique.




      N. Boussaha ()                Processus Aléatoires (1)   USTHB, 2024-2025   37 / 66
</pre>

</details>
<details>
<summary>PDF page 64</summary>

<pre>
Distribution stationnaire




Example
cas de deux classes irréductibles fermées :
                                               Remarque
                                               pour cet exemple :
                                                  Il existe une infinité de
                                               distributions stationnaires.
                                                  La chaîne contient plus
                                               d’une classe récurrente.




      N. Boussaha ()             Processus Aléatoires (1)         USTHB, 2024-2025   38 / 66
</pre>

</details>
<details>
<summary>PDF page 65</summary>

<pre>
Distribution stationnaire




Example
                         1   1    1      1         1         1   1
S = N, avec            0 ! 1 ! 2 ! 3 ! ... ! n ! n + 1 ! ...

Remarque
pour cet exemple :
  Il n’existe pas de distribution stationnaire.
  La chaîne est transitoire.



      N. Boussaha ()              Processus Aléatoires (1)           USTHB, 2024-2025   39 / 66
</pre>

</details>
<details>
<summary>PDF page 66</summary>

<pre>
Distribution stationnaire




Example
                            1        2/3       2/3         2/3       2/3         2/3
S = N, avec            0         1         2         ...         n         n+1         ...
                           1/3       1/3       1/3         1/3       1/3         1/3

Question : Qu’en est-il de cette Chaîne ?



      N. Boussaha ()                           Processus Aléatoires (1)                      USTHB, 2024-2025   40 / 66
</pre>

</details>
<details>
<summary>PDF page 67</summary>

<pre>
Distribution stationnaire




Example
                           1        p1       p2         pn 1        pn            p n +1
S = N, avec            0        1        2        ...          n            n+1            ...
                           q1       q2       q3          qn        q n +1         q n +2

Question : Qu’en est-il de cette Chaîne ? et si pi = p et qi = q, 8i                                            1.



      N. Boussaha ()                          Processus Aléatoires (1)                       USTHB, 2024-2025   41 / 66
</pre>

</details>
<details>
<summary>PDF page 68</summary>

<pre>
Distribution stationnaire

Theorem
Si une chaîne de Markov possède une distribution stationnaire π, alors
pour tout état j transitoire ou récurrent nul : π j = 0.




      N. Boussaha ()          Processus Aléatoires (1)   USTHB, 2024-2025   42 / 66
</pre>

</details>
<details>
<summary>PDF page 69</summary>

<pre>
Distribution stationnaire

Theorem
Si une chaîne de Markov possède une distribution stationnaire π, alors
pour tout état j transitoire ou récurrent nul : π j = 0.


Démonstration.
                                                         (n )
Puisque j est transitoire, on a limn !∞ pij                     = 0 (8i 2 S). De plus, comme
                 (n )
π j = ∑i 2S π i pij , il en résulte que :

                                                          =1          !0
                                                        z }| {z }| {
                       π j = lim ∑                      ∑ πi lim pij = 0.
                                              (n )                (n )
                                         π i pij =
                           n !∞                                     n !∞
                                  i 2S                  i 2S
                           |                       {z                      }
                               Th éor ème de convergence domin ée


      N. Boussaha ()                     Processus Aléatoires (1)              USTHB, 2024-2025   43 / 66
</pre>

</details>
<details>
<summary>PDF page 70</summary>

<pre>
Distribution stationnaire




Questions :

    Quelles sont les conditions de l’existence et de l’unicité d’une
    distribution stationnaire ?




     N. Boussaha ()           Processus Aléatoires (1)    USTHB, 2024-2025   44 / 66
</pre>

</details>
<details>
<summary>PDF page 71</summary>

<pre>
Distribution stationnaire




Questions :

    Quelles sont les conditions de l’existence et de l’unicité d’une
    distribution stationnaire ?

    Est-ce que la distribution stationnaire a une interprétation pratique ?




     N. Boussaha ()           Processus Aléatoires (1)    USTHB, 2024-2025   44 / 66
</pre>

</details>
<details>
<summary>PDF page 72</summary>

<pre>
Distribution stationnaire
Theorem (Relation entre π j et µj )
Si une Chaîne de Markov est irréductible et de récurrence positive alors il
existe une distribution stationnaire unique π, donnée par
                                   1
                           πj =       ,       8j 2 S.
                                   µj



      N. Boussaha ()          Processus Aléatoires (1)   USTHB, 2024-2025   45 / 66
</pre>

</details>
