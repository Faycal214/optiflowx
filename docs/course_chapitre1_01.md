<details>
<summary>PDF page 1</summary>

<pre>
                 Processus aléatoires

Master Modélisation Stochastique et Prévision en Recherche
                Opérationnelle (MSPRO)
Chapitre 1 : Chaînes de Markov à Temps Discret (CMTD)



                   USTHB, 2024-2025




N. Boussaha ()       Processus Aléatoires (1)   USTHB, 2024-2025   1 / 66
</pre>

</details>
<details>
<summary>PDF page 2</summary>

<pre>
Introduction : Processus aléatoires


    Un processus aléatoire est une collection de variables aléatoires
    indexées par le temps ou un autre paramètre, modélisant des
    phénomènes évoluant de manière incertaine.




     N. Boussaha ()           Processus Aléatoires (1)    USTHB, 2024-2025   2 / 66
</pre>

</details>
<details>
<summary>PDF page 3</summary>

<pre>
Introduction : Processus aléatoires


    Un processus aléatoire est une collection de variables aléatoires
    indexées par le temps ou un autre paramètre, modélisant des
    phénomènes évoluant de manière incertaine.

    Chaque état du système est soumis à une certaine probabilité et
    évolue selon des règles de transition aléatoires.




     N. Boussaha ()           Processus Aléatoires (1)    USTHB, 2024-2025   2 / 66
</pre>

</details>
<details>
<summary>PDF page 4</summary>

<pre>
Introduction : Processus aléatoires


    Un processus aléatoire est une collection de variables aléatoires
    indexées par le temps ou un autre paramètre, modélisant des
    phénomènes évoluant de manière incertaine.

    Chaque état du système est soumis à une certaine probabilité et
    évolue selon des règles de transition aléatoires.

    Les processus aléatoires sont utilisés pour modéliser une grande
    variété de systèmes, comme




     N. Boussaha ()           Processus Aléatoires (1)    USTHB, 2024-2025   2 / 66
</pre>

</details>
<details>
<summary>PDF page 5</summary>

<pre>
Introduction : Processus aléatoires


    Un processus aléatoire est une collection de variables aléatoires
    indexées par le temps ou un autre paramètre, modélisant des
    phénomènes évoluant de manière incertaine.

    Chaque état du système est soumis à une certaine probabilité et
    évolue selon des règles de transition aléatoires.

    Les processus aléatoires sont utilisés pour modéliser une grande
    variété de systèmes, comme
          les files d’attente,




     N. Boussaha ()             Processus Aléatoires (1)   USTHB, 2024-2025   2 / 66
</pre>

</details>
<details>
<summary>PDF page 6</summary>

<pre>
Introduction : Processus aléatoires


    Un processus aléatoire est une collection de variables aléatoires
    indexées par le temps ou un autre paramètre, modélisant des
    phénomènes évoluant de manière incertaine.

    Chaque état du système est soumis à une certaine probabilité et
    évolue selon des règles de transition aléatoires.

    Les processus aléatoires sont utilisés pour modéliser une grande
    variété de systèmes, comme
          les files d’attente,
          la propagation d’épidémies,




     N. Boussaha ()             Processus Aléatoires (1)   USTHB, 2024-2025   2 / 66
</pre>

</details>
<details>
<summary>PDF page 7</summary>

<pre>
Introduction : Processus aléatoires


    Un processus aléatoire est une collection de variables aléatoires
    indexées par le temps ou un autre paramètre, modélisant des
    phénomènes évoluant de manière incertaine.

    Chaque état du système est soumis à une certaine probabilité et
    évolue selon des règles de transition aléatoires.

    Les processus aléatoires sont utilisés pour modéliser une grande
    variété de systèmes, comme
          les files d’attente,
          la propagation d’épidémies,
          les variations de cours boursiers...




     N. Boussaha ()               Processus Aléatoires (1)   USTHB, 2024-2025   2 / 66
</pre>

</details>
<details>
<summary>PDF page 8</summary>

<pre>
Introduction : Processus aléatoires

On considère (Ω, A, P) un espace de probabilité, (T , T ) un espace
mesurable. On prend une application :

                 X :Ω     T ! E,       (w , t ) 7! X (w , t ) = Xt (w )

telle que Xt (w ) soit une variable aléatoire sur (Ω, A) pour tout t 2 T .
On adaptera la notation suivante :

                  fX (t ), t 2 T g = fX (w , t ), w 2 Ω et t 2 T g

La famille fX (t ), t 2 T g est appelée processus stochastique, ou processus
aléatoire.
Un processus stochastique, ou processus aléatoire, est une famille de
variables aléatoires fX (t ), t 2 T g. Ces variables aléatoires sont définies
sur un même espace de probabilité (Ω, A, P).

      N. Boussaha ()              Processus Aléatoires (1)        USTHB, 2024-2025   3 / 66
</pre>

</details>
<details>
<summary>PDF page 9</summary>

<pre>
Introduction : Processus aléatoires




 Temps T       Espace des états S                                            Exemple de
  Discret           Discret                                  Chaîne de Markov à temps d
  Discret           Continu                                                Modèle de série
 Continu            Discret                            Processus de sauts (par exemple, p
 Continu            Continu           Processus de Wiener (mouvement brownien) ou tout a

Tab.: Classification des processus stochastiques selon le temps et l’espace des
états




      N. Boussaha ()                Processus Aléatoires (1)      USTHB, 2024-2025   4 / 66
</pre>

</details>
<details>
<summary>PDF page 10</summary>

<pre>
Chaînes de Markov homogenes

Definition
Une chaîne de Markov est un processus stochastique à temps discret et
espace d’états discret qui vérifie la propriété de Marov

P(Xn +1 = j j X0 = i0 , . . . , Xn 1 = in 1 , Xn = i ) = P(Xn +1 = j j Xn = i )

8 n 2 T = N et i0 , . . . , in 1 , i, j 2 S.


Definition
Une chaîne de Markov est dite homogène si

    P(Xn +1 = j j Xn = i ) = P(X1 = j j X0 = i )              8 n 2 T et i, j 2 S.

On écrit pi ,j = P(Xn +1 = j j Xn = i )..


       N. Boussaha ()              Processus Aléatoires (1)         USTHB, 2024-2025   5 / 66
</pre>

</details>
<details>
<summary>PDF page 11</summary>

<pre>
Matrice de transition


On peut regrouper les probabilités de transition d’une CMH sous forme
matricielle
                             0                                 1
                               p00 p01 p02              p0j
                             B ..  ..                          C
                             B .      .                        C
                             B                                 C
                             B
       P = (pi ,j )i ,j 2S = B                pii       pij    C.
                                                               C
                             B                 ..  ..       .. C
                                                            .
                             @                  .     .        A



    La matrice P est une matrice stochastique : pij     0 8 i, j 2 S et
    ∑j 2S pij = 1 8 i 2 S.



      N. Boussaha ()         Processus Aléatoires (1)     USTHB, 2024-2025   6 / 66
</pre>

</details>
<details>
<summary>PDF page 12</summary>

<pre>
Graphe associé




Le graphe associé à la matrice de transition d’une chaîne de Markov est
un graphe orienté où chaque état de la chaîne est représenté par un nœud,
et chaque transition possible entre états est représentée par une arête
dirigée. Le poids de chaque arête correspond à la probabilité de transition
entre les deux états associés.




      N. Boussaha ()          Processus Aléatoires (1)   USTHB, 2024-2025   7 / 66
</pre>

</details>
<details>
<summary>PDF page 13</summary>

<pre>
Example
Une marche aléatoire sur l’espace d’états S = Z est une chaîne de
Markov fXn , n 2 Ng qui vérifie les probabilités de transitions suivantes :
                                           8
                                           >
                                           >  p si j = i + 1,
                                           >
                                           >
                                           <q si j = i 1,
            pij = P(Xn +1 = j j Xn = i ) =
                                           >
                                           >  r si j = i,
                                           >
                                           >
                                           :0 sinon.

avec p + q + r = 1.

Figure



      N. Boussaha ()          Processus Aléatoires (1)    USTHB, 2024-2025   8 / 66
</pre>

</details>
<details>
<summary>PDF page 14</summary>

<pre>
Caractérisation d une Chaine de Markov
   la loi initiale de la chaîne de Markov est
                           µ0 = (P(X0 = i ))i 2S .




    N. Boussaha ()           Processus Aléatoires (1)   USTHB, 2024-2025   9 / 66
</pre>

</details>
<details>
<summary>PDF page 15</summary>

<pre>
Caractérisation d une Chaine de Markov
   la loi initiale de la chaîne de Markov est
                           µ0 = (P(X0 = i ))i 2S .
   Une chaîne de Markov est entièrement caractérisée par la matrice de
   transition P et la loi initiale µ0 .




    N. Boussaha ()           Processus Aléatoires (1)   USTHB, 2024-2025   9 / 66
</pre>

</details>
<details>
<summary>PDF page 16</summary>

<pre>
Caractérisation d une Chaine de Markov
   la loi initiale de la chaîne de Markov est
                                 µ0 = (P(X0 = i ))i 2S .
   Une chaîne de Markov est entièrement caractérisée par la matrice de
   transition P et la loi initiale µ0 .
   En effet, pour tout i0 , i1 , ..., in 2 S
                P(X0 = i0 , X1 = i1 , . . . , Xn = in )
                         = P(Xn = in j X0 = i0 , . . . , Xn 1 = in 1 )
                      P(X0 = i0 , . . . , Xn 1 = in 1 )
                         = P(Xn = in j Xn 1 = in 1 )
                      P(Xn 1 = in 1 j X0 = i0 , . . . , Xn 2 = in 2 )
                      P(X0 = i0 , . . . , Xn 2 = in 2 )
                      ..
                       .
                         = P(X0 = i )pi0 i1 pi1 i2 pin 1 in .
    N. Boussaha ()                 Processus Aléatoires (1)    USTHB, 2024-2025   9 / 66
</pre>

</details>
<details>
<summary>PDF page 17</summary>

<pre>
Matrice de transition en n étapes



    On note la probabilité de passage de l’état i à l’état j en n transitions
    par :
                         (n )
                       pij = P(Xn +m = j j Xm = i ),
    pour m, n 2 T et i, j 2 S.



     N. Boussaha ()           Processus Aléatoires (1)    USTHB, 2024-2025   10 / 66
</pre>

</details>
<details>
<summary>PDF page 18</summary>

<pre>
Matrice de transition en n étapes



    On note la probabilité de passage de l’état i à l’état j en n transitions
    par :
                         (n )
                       pij = P(Xn +m = j j Xm = i ),
    pour m, n 2 T et i, j 2 S.

    La matrice de transtion en n étapes :
                                               (n )
                              P(n ) = pij                          .
                                                         i ,j 2S



     N. Boussaha ()           Processus Aléatoires (1)                 USTHB, 2024-2025   10 / 66
</pre>

</details>
