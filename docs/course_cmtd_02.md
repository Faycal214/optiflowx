<details>
<summary>Course slide 23</summary>

<pre>
Un autre critere de classification des états

Lemma
Un état i est récurrent si et seulement si ∑n∞=1 pii
                                                           (n )
                                                                  = ∞ et il est
transitoire si et seulement si ∑n∞=1 pii &lt; ∞.
                                      (n )


Example
Marche aléatoire sur Z (TD).

Proposition
Si i $ j et si i est récurrent, alors j est aussi récurrent. On dit que la
récurrence est une propriété de classe

Démonstration.
(TD). Indication : Penser à utiliser les équations de
Chapman-Kolmogorov.
      N. Boussaha ()            Processus Aléatoires (1)             USTHB, 2024-2025   23 / 66
</pre>

</details>
<details>
<summary>Course slide 24</summary>

<pre>
Temps moyen de retour
Definitions
    Tij : temps d’atteinte de l’état j pour la première fois à partir de i
    et après n transitions est défini par :
                      Tij = minfn         1 : Xn = j j X0 = i g.

    On retrouve une expression de la probabilité de première visite de
    l’état j à partir de i en n transitions par
                               (n )
                             fij      = P (Tij = n) .

    mij : le temps moyen de visite de j étant initialement à i
                                      ∞                               ∞
       µij = E (Tij j X0 = i ) = ∑ nP(Tij = n j X0 = j ) = ∑ nfjj .
                                                                             (n )

                                   n =1                              n =1

    On peut aussi définir µj = E (Tjj j X0 = j ) comme temps moyen de
    retour à l’état j.
     N. Boussaha ()           Processus Aléatoires (1)        USTHB, 2024-2025      24 / 66
</pre>

</details>
<details>
<summary>Course slide 25</summary>

<pre>
Temps moyen de retour
                                              (n )                  (n )
Theorem (Relation entre les (pij ) et les (fij ))
On a :
                                n
                             = ∑ fij pjj
                      (n )            (k ) (n k )
                  pij                                ,     8i, j 2 S et n    1.
                               k =1




     N. Boussaha ()                      Processus Aléatoires (1)           USTHB, 2024-2025   25 / 66
</pre>

</details>
<details>
<summary>Course slide 26</summary>

<pre>
Temps moyen de retour
                                                (n )                  (n )
Theorem (Relation entre les (pij ) et les (fij ))
On a :
                                  n
                               = ∑ fij pjj
                        (n )            (k ) (n k )
                  pij                                ,     8i, j 2 S et n    1.
                                 k =1

Explication :
       8
              (1 ) (3 )
       &gt;
       &gt;   fij pjj
       &gt;
       &gt;                       correspond aux chemins où on atteint j immédiatement (en
       &gt;
       &gt;                       1 étape), puis on effectue un chemin de 3 étapes en revenant à
       &gt;
       &gt;
       &gt;
       &gt;   +
       &gt;
       &gt;      (2 ) (2 )
       &gt;
       &gt;   fij pjj
                               correspond aux chemins où on atteint j pour la 1ere fois en 2 étape
       &gt;
       &lt;
 (4 )                          avant d’effectuer un chemin de 2 étapes en revenant à j
pij =            +
       &gt;
       &gt;      (3 ) (1 )
       &gt;
       &gt;   f ij pjj            correspond aux chemins où on atteint j pour la 1ere fois en 3 étape
       &gt;
       &gt;
       &gt;
       &gt;                       avant d’effectuer la dernière transition qui est de revenir à j
       &gt;
       &gt;         +
       &gt;
       &gt;
       &gt;
       &gt;   f
              (4 ) (0 )
                  p            correspond aux chemins où j est atteint pour la première fois
       &gt;
       : ij jj                 en 4 étapes, ce qui est égal au nombre total de transitions
       N. Boussaha ()                      Processus Aléatoires (1)           USTHB, 2024-2025   26 / 66
</pre>

</details>
<details>
<summary>Course slide 27</summary>

<pre>
Temps moyen de retour


Démonstration.
   Soient (i, j ) 2 S    S, n 2 N et k = 1, . . . , n. On définit

                        Ek = fXn = j, Ti ,j = k j X0 = i g.
    Il est clair que pour tout k 6= k 0 , on a Ek \ Ek 0 = ∅.
    Donc
                                                     !
                                          n
                                          [                  n             n
                                                          = ∑ P(Ek ) = ∑ fi ,j pj ,j
     (n )                                                                       (k ) (n k )
    pi ,j = P(Xn = j j X0 = i ) = P             Ek                                            .
                                         k =1               k =1         k =1



      N. Boussaha ()           Processus Aléatoires (1)              USTHB, 2024-2025   27 / 66
</pre>

</details>
<details>
<summary>Course slide 28</summary>

<pre>
Proposition
Soit j un état transitoire. Alors, pour tout i 2 S
                                     (n )
                                lim p = 0.
                               n !∞ ij



      N. Boussaha ()           Processus Aléatoires (1)   USTHB, 2024-2025   28 / 66
</pre>

</details>
<details>
<summary>Course slide 29</summary>

<pre>
Proposition
Soit j un état transitoire. Alors, pour tout i 2 S
                                                (n )
                                           lim p = 0.
                                          n !∞ ij


Démonstration.
                                         (n )
  Pour i = j : on a limn !∞ pjj                 = 0, car
                  j transitoire , ∑n∞=1 pjj
                                                     (n )                      (n )
                                                            &lt; ∞ ) limn !∞ pjj = 0.
  Pour i 6= j :
                                          n                          n
                              lim ∑ fij pjj                       = ∑ lim fij pjj
                   (n )                         (k ) (n k )                 (k ) (n k )
           lim pij        =
          n !∞                n !∞                                        n !∞
                                       k =1
                              |                                   {z k =1              }
                                          Th éor ème de convergence domin ée
                                  n
                              ∑ fij nlim
                                        (k )           (n k )
                          =              p                        = 0.
                                      !∞ jj
                              | {z }|
                              k =1                  {z
                                                    !0
                                                              }
                                  fij 2[0,1 ]
      N. Boussaha ()                      Processus Aléatoires (1)           USTHB, 2024-2025   29 / 66
</pre>

</details>
<details>
<summary>Course slide 30</summary>

<pre>
Récurrence nulle et récurrence positive



Definition
  Un état de récurrence j est récurrent positif si µj &lt; ∞.
  Il est dit récurrent nul si µj = ∞.

      Les récurrences positive et nulle sont des propriétés de classe.
Démonstration.
TD.



       N. Boussaha ()           Processus Aléatoires (1)    USTHB, 2024-2025   30 / 66
</pre>

</details>
<details>
<summary>Course slide 31</summary>

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
<summary>Course slide 32</summary>

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
<summary>Course slide 33</summary>

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
<summary>Course slide 34</summary>

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
<summary>Course slide 35</summary>

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
       N. Boussaha ()             Processus Aléatoires (1)   USTHB, 2024-2025   35 / 66
</pre>

</details>
<details>
<summary>Course slide 36</summary>

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



      N. Boussaha ()           Processus Aléatoires (1)   USTHB, 2024-2025   36 / 66
</pre>

</details>
<details>
<summary>Course slide 37</summary>

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
<summary>Course slide 38</summary>

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
<summary>Course slide 39</summary>

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
<summary>Course slide 40</summary>

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
<summary>Course slide 41</summary>

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
<summary>Course slide 42</summary>

<pre>
Distribution stationnaire

Theorem
Si une chaîne de Markov possède une distribution stationnaire π, alors
pour tout état j transitoire ou récurrent nul : π j = 0.




      N. Boussaha ()          Processus Aléatoires (1)   USTHB, 2024-2025   42 / 66
</pre>

</details>
<details>
<summary>Course slide 43</summary>

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
<summary>Course slide 44</summary>

<pre>
Distribution stationnaire




Questions :

    Quelles sont les conditions de l’existence et de l’unicité d’une
    distribution stationnaire ?

    Est-ce que la distribution stationnaire a une interprétation pratique ?




     N. Boussaha ()           Processus Aléatoires (1)    USTHB, 2024-2025   44 / 66
</pre>

</details>
