<details>
<summary>Course slide 1</summary>

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
<summary>Course slide 2</summary>

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
<summary>Course slide 3</summary>

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
<summary>Course slide 4</summary>

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
<summary>Course slide 5</summary>

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
<summary>Course slide 6</summary>

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
<summary>Course slide 7</summary>

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
<summary>Course slide 8</summary>

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
<summary>Course slide 9</summary>

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
<summary>Course slide 10</summary>

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

    Proposition : P(n ) est une matrice stochastique.




     N. Boussaha ()           Processus Aléatoires (1)                 USTHB, 2024-2025   10 / 66
</pre>

</details>
<details>
<summary>Course slide 11</summary>

<pre>
Equations de Chapman-Kolmogorov



Proposition
On a
                                     = ∑ pik pkj ,
                           (n +m )               (m ) (n )
                         pij
                                        k 2S

pour tout m, n 2 N et i, j 2 S.




       N. Boussaha ()          Processus Aléatoires (1)      USTHB, 2024-2025   11 / 66
</pre>

</details>
<details>
<summary>Course slide 12</summary>

<pre>
Equations de Chapman-Kolmogorov
Démonstration.
On considère une chaîne de Markov homogène (C.M.H) fXn , n 2 Ng,
donc
     (n +m )
    pij        = P(Xn +m = j j X0 = i )
               = P(Xn +m = j, [k 2S fXm = k g j X0 = i )
               = ∑ P(Xn +m = j, Xm = k j X0 = i )
                  k 2S
               = ∑ P(Xn +m = j, Xm = k, X0 = i )P(Xm = k j X0 = i )
                  k 2S
               = ∑ P(Xn +m = j, Xm = k, X0 = i )P(Xm = k j X0 = i )
                  k 2S

               = ∑ pkj pik .
                         (n ) (m )

                  k 2S



     N. Boussaha ()                  Processus Aléatoires (1)   USTHB, 2024-2025   12 / 66
</pre>

</details>
<details>
<summary>Course slide 13</summary>

<pre>
Equations de Chapman-Kolmogorov


Remarque
Les équations de Chapman-Kolmogorov sous forme matricielle

                             P(m +n ) = P(m )             P(n ) .

Ce qui implique
                        P(n ) = P
                                |
                                  (1 )
                                           P(1 ){z              P(1}) .
                                                 n fois

Comme P(1 ) = P, on a :
                                     P(n ) = Pn .
     (n )
    pij     6= pijn .



     N. Boussaha ()                Processus Aléatoires (1)               USTHB, 2024-2025   13 / 66
</pre>

</details>
<details>
<summary>Course slide 14</summary>

<pre>
Récurrences markoviennes


Theorem
Soit fξ n gn 1 une suite i.i.d de v .a. à valeurs dans un espace arbitraire G.
Soit S un espace dénombrable, et f : S G ! S. Soit X0 une v .a. à
valeurs dans S, indépendante de fξ n gn 1 . Le processus X = fXn , n 2 Ng
exprimée par la relation de récurrence :

                            Xn +1 = f (Xn , ξ n +1 ),

définit alors une chaîne de Markov homogène (CMH).

Démonstration.
TD.



      N. Boussaha ()            Processus Aléatoires (1)   USTHB, 2024-2025   14 / 66
</pre>

</details>
<details>
<summary>Course slide 15</summary>

<pre>
Distribution de létat d une chaine de Markov


    On note par µn = (P(Xn = i ))i 2S (vecteur ligne) la loi de
    distribution de Xn .

    D’après la formule des probabilités totales, on a pour un état j 2 S

                 P(Xn   = j ) = ∑ P(Xn = j j X0 = i )P(X0 = i )
                                  i 2S
                             (n )
                        =   pij µ0 (i ) , (si S = N )

          ce qui donne sous forme matricielle la formule

                                          µn = µ0 Pn .



     N. Boussaha ()             Processus Aléatoires (1)   USTHB, 2024-2025   15 / 66
</pre>

</details>
<details>
<summary>Course slide 16</summary>

<pre>
Probabilité de premiere visite
Definition
La probabilité de première visite d’un état j en n transitions ou étapes,
partant d’un état i est donnée par
                  (n )
                fij      = P (Xn = j, Xn 1 6= j, . . . , X1 6= j j X0 = i ) ,

pour n        1 et i, j 2 S.
       (n )
     fii      : la probabilité de premier retour à i en n transitions.

Definition
La probabilité que partant de i on visite l’état j par :
                                                   ∞
                                        fij = ∑ fij .
                                                          (n )

                                                 n =1

     fii : la probabilité de retour à l’état i.
       N. Boussaha ()                   Processus Aléatoires (1)      USTHB, 2024-2025   16 / 66
</pre>

</details>
<details>
<summary>Course slide 17</summary>

<pre>
Relation de communication

Notation
    On dit que l’état j est accessible de l’état i s’il existe un n                0 tel
         (n )
    que pij > 0.
    Si les deux états i et j sont accessibles l’un à l’autre alors on dit qu’ils
    communiquent.
    On note la relation de communication entre deux états i et j par :
    i $ j.
    La relation de communication " $ " est une relation d’équivalence.
    En effet :
                                   (0 )
           Réflexive : i $ i car pii       = 1 > 0 pour tout i 2 S
             (0 )
           (pii = P(X0 = i j X0 = i ) = 1)
           Symétrique : Si i $ j alors j $ i pour tout i, j 2 S
           Transitive : Si i $ j et j $ k alors i $ k pour i, j, k 2 S.

     N. Boussaha ()               Processus Aléatoires (1)          USTHB, 2024-2025   17 / 66
</pre>

</details>
<details>
<summary>Course slide 18</summary>

<pre>
Relation de communication


Démonstration.
(Transitivité)
On
( a:                           (n )
   i $ j =) i ! j =) 9n 2 N , pij > 0,
                                                   (m )
   j $ k =) j ! k =) 9n 2 N , pjk > 0,
En utilisant les équations de Chapman-Kolmogorov :

                          = ∑ pil plk
                (n +m )            (m ) (n )         (n ) (m )
              pik                                  pij pjk        > 0 =) i ! k.
                            l 2S

De même, on montre que si i $ j et j $ k alors k ! i.



     N. Boussaha ()                    Processus Aléatoires (1)           USTHB, 2024-2025   18 / 66
</pre>

</details>
<details>
<summary>Course slide 19</summary>

<pre>
Relation de communication



Remarque
Les classes de communication forme une partition de l’espace des états S.



      N. Boussaha ()          Processus Aléatoires (1)   USTHB, 2024-2025   19 / 66
</pre>

</details>
<details>
<summary>Course slide 20</summary>

<pre>
Recurrences markoviennes

Definition
On dit qu’une Chaîne de Markov est irréductible si l’espace des états S ne
forme qu’une seule classe de communication.

Definition
Une classe C est dite absorbante ou fermée si

                       P(Xn +1 2 C j Xn = i ) = 1 8i 2 C .

ou d’une façon équivalente : Une classe C est dite fermée, si 8i 2 C ,
                    (n )
   / C et 8n 2 S : pij = 0.
8j 2

Remarque
Un état i est dit absorbant si pii = 1.


      N. Boussaha ()             Processus Aléatoires (1)    USTHB, 2024-2025   20 / 66
</pre>

</details>
<details>
<summary>Course slide 21</summary>

<pre>
Propriété de Récurrence et transience


Definition
   Un état j est dit récurrent si, partant de cet état, on y revient presque
sûrement, i.e. fjj = 1.
   Si fjj < 1 alors j est un état transient.

Definition
Une Chaîne de Markov est transitoire si tous ses états sont transitoires, et
on dit qu’elle récurrente si tous ses états sont récurrents.

Example
CMH avec un état absorbant.



      N. Boussaha ()           Processus Aléatoires (1)   USTHB, 2024-2025   21 / 66
</pre>

</details>
<details>
<summary>Course slide 22</summary>

<pre>
Nombre moyen de visites
Definition
On définit N (i, i ) le nombre de visites à l’état i partant de i
                                                 ∞
                               N (i, i ) = ∑ 1X n =i jX 0 =i .
                                                n =1

                                                                                                (n )
Comme E (1X n =i j X0 = i ) = E 1X n =i jX 0 =i = P (Xn = i j X0 = i ) = pii ,

                                                     !
                                ∞                                  ∞                    ∞
                               ∑ 1Xn =i jX0 =i            = ∑ E 1Xn =i jX0 =i = ∑ pii ,
                                                                                              (n )
 alors E (N (i, i )) = E
                               n =1                           n =1                     n =1
                         |                              {z                       }
                                    Th éor ème de convergence monotone

représente le nombre moyen de retours à l’état i , à long terme.

Ce nombre moyen sera donc infini si l’état i est récurrent puisque on y revient, presque
sûrement, indéfiniment à long terme. Ce qui nous donne la caractérisation suivante des
états récurrents et transitoires.
       N. Boussaha ()                   Processus Aléatoires (1)           USTHB, 2024-2025    22 / 66
</pre>

</details>
