<details>
<summary>PDF page 19</summary>

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
<summary>PDF page 20</summary>

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
<summary>PDF page 21</summary>

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
<summary>PDF page 22</summary>

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
<summary>PDF page 23</summary>

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
<summary>PDF page 24</summary>

<pre>
Distribution de létat d une chaine de Markov


    On note par µn = (P(Xn = i ))i 2S (vecteur ligne) la loi de
distribution de Xn .




     N. Boussaha ()          Processus Aléatoires (1)  USTHB, 2024-2025   15 / 66
</pre>

</details>
<details>
<summary>PDF page 25</summary>

<pre>
Distribution de létat d une chaine de Markov


    On note par µn = (P(Xn = i ))i 2S (vecteur ligne) la loi de
distribution de Xn .

    D’après la formule des probabilités totales, on a pour un état j 2 S

                 P(Xn   = j ) = ∑ P(Xn = j j X0 = i )P(X0 = i )
                                  i 2S
                             (n )
                        =   pij µ0 (i ) , (si S = N )



     N. Boussaha ()             Processus Aléatoires (1)   USTHB, 2024-2025   15 / 66
</pre>

</details>
<details>
<summary>PDF page 26</summary>

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
<summary>PDF page 27</summary>

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



       N. Boussaha ()                   Processus Aléatoires (1)      USTHB, 2024-2025   16 / 66
</pre>

</details>
<details>
<summary>PDF page 28</summary>

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
<summary>PDF page 29</summary>

<pre>
Relation de communication

Notation
    On dit que l’état j est accessible de l’état i s’il existe un n       0 tel
         (n )
    que pij > 0.



     N. Boussaha ()            Processus Aléatoires (1)    USTHB, 2024-2025   17 / 66
</pre>

</details>
<details>
<summary>PDF page 30</summary>

<pre>
Relation de communication

Notation
    On dit que l’état j est accessible de l’état i s’il existe un n        0 tel
         (n )
    que pij > 0.
    Si les deux états i et j sont accessibles l’un à l’autre alors on dit qu’ils
    communiquent.



     N. Boussaha ()            Processus Aléatoires (1)     USTHB, 2024-2025   17 / 66
</pre>

</details>
<details>
<summary>PDF page 31</summary>

<pre>
Relation de communication

Notation
    On dit que l’état j est accessible de l’état i s’il existe un n        0 tel
         (n )
    que pij > 0.
    Si les deux états i et j sont accessibles l’un à l’autre alors on dit qu’ils
    communiquent.
    On note la relation de communication entre deux états i et j par :
    i $ j.



     N. Boussaha ()            Processus Aléatoires (1)     USTHB, 2024-2025   17 / 66
</pre>

</details>
<details>
<summary>PDF page 32</summary>

<pre>
Relation de communication

Notation
    On dit que l’état j est accessible de l’état i s’il existe un n        0 tel
         (n )
    que pij > 0.
    Si les deux états i et j sont accessibles l’un à l’autre alors on dit qu’ils
    communiquent.
    On note la relation de communication entre deux états i et j par :
    i $ j.
    La relation de communication "$ " est une relation d’équivalence.
    En effet :



     N. Boussaha ()            Processus Aléatoires (1)     USTHB, 2024-2025   17 / 66
</pre>

</details>
<details>
<summary>PDF page 33</summary>

<pre>
Relation de communication

Notation
    On dit que l’état j est accessible de l’état i s’il existe un n               0 tel
         (n )
    que pij > 0.
    Si les deux états i et j sont accessibles l’un à l’autre alors on dit qu’ils
    communiquent.
    On note la relation de communication entre deux états i et j par :
    i $ j.
    La relation de communication "$ " est une relation d’équivalence.
    En effet :
                                  (0 )
           Réflexive : i $ i car pii      = 1 > 0 pour tout i 2 S
             (0 )
           (pii = P(X0 = i j X0 = i ) = 1)



     N. Boussaha ()              Processus Aléatoires (1)          USTHB, 2024-2025   17 / 66
</pre>

</details>
<details>
<summary>PDF page 34</summary>

<pre>
Relation de communication

Notation
    On dit que l’état j est accessible de l’état i s’il existe un n               0 tel
         (n )
    que pij > 0.
    Si les deux états i et j sont accessibles l’un à l’autre alors on dit qu’ils
    communiquent.
    On note la relation de communication entre deux états i et j par :
    i $ j.
    La relation de communication "$ " est une relation d’équivalence.
    En effet :
                                  (0 )
           Réflexive : i $ i car pii      = 1 > 0 pour tout i 2 S
             (0 )
           (pii = P(X0 = i j X0 = i ) = 1)
           Symétrique : Si i $ j alors j $ i pour tout i, j 2 S


     N. Boussaha ()              Processus Aléatoires (1)          USTHB, 2024-2025   17 / 66
</pre>

</details>
<details>
<summary>PDF page 35</summary>

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
    La relation de communication "$ " est une relation d’équivalence.
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
<summary>PDF page 36</summary>

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
