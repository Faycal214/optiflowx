<details>
<summary>PDF page 37</summary>

<pre>
Relation de communication




Remarque
Les classes de communication forme une partition de l’espace des états S.



      N. Boussaha ()          Processus Aléatoires (1)   USTHB, 2024-2025   19 / 66
</pre>

</details>
<details>
<summary>PDF page 38</summary>

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


      N. Boussaha ()             Processus Aléatoires (1)   USTHB, 2024-2025   20 / 66
</pre>

</details>
<details>
<summary>PDF page 39</summary>

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
<summary>PDF page 40</summary>

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
<details>
<summary>PDF page 41</summary>

<pre>
Un autre critere de classification des états


Lemma
Un état i est récurrent si et seulement si ∑n∞=1 pii
                                                           (n )
                                                                  = ∞ et il est
transitoire si et seulement si ∑n∞=1 pii < ∞.
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
<summary>PDF page 42</summary>

<pre>
Temps moyen de retour
Definitions
    Tij : temps d’atteinte de l’état j pour la première fois à partir de i
    et après n transitions est défini par :
                      Tij = minfn        1 : Xn = j j X0 = i g.



     N. Boussaha ()           Processus Aléatoires (1)       USTHB, 2024-2025   24 / 66
</pre>

</details>
<details>
<summary>PDF page 43</summary>

<pre>
Temps moyen de retour
Definitions
    Tij : temps d’atteinte de l’état j pour la première fois à partir de i
    et après n transitions est défini par :
                      Tij = minfn        1 : Xn = j j X0 = i g.

    On retrouve une expression de la probabilité de première visite de
    l’état j à partir de i en n transitions par
                               (n )
                             fij      = P (Tij = n) .



     N. Boussaha ()           Processus Aléatoires (1)       USTHB, 2024-2025   24 / 66
</pre>

</details>
<details>
<summary>PDF page 44</summary>

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



     N. Boussaha ()           Processus Aléatoires (1)        USTHB, 2024-2025      24 / 66
</pre>

</details>
<details>
<summary>PDF page 45</summary>

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
<summary>PDF page 46</summary>

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
<summary>PDF page 47</summary>

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
       >
       >   fij pjj
       >
       >                       correspond aux chemins où on atteint j immédiatement (en
       >
       >                       1 étape), puis on effectue un chemin de 3 étapes en revenant à
       >
       >
       >
       >   +
       >
       >      (2 ) (2 )
       >
       >   fij pjj
                               correspond aux chemins où on atteint j pour la 1ere fois en 2 étape
       >
       <
 (4 )                          avant d’effectuer un chemin de 2 étapes en revenant à j
pij =            +
       >
       >      (3 ) (1 )
       >
       >   f ij pjj            correspond aux chemins où on atteint j pour la 1ere fois en 3 étape
       >
       >
       >
       >                       avant d’effectuer la dernière transition qui est de revenir à j
       >
       >         +
       >
       >
       >
       >   f
              (4 ) (0 )
                  p            correspond aux chemins où j est atteint pour la première fois
       >
       : ij jj                 en 4 étapes, ce qui est égal au nombre total de transitions
       N. Boussaha ()                      Processus Aléatoires (1)           USTHB, 2024-2025   26 / 66
</pre>

</details>
<details>
<summary>PDF page 48</summary>

<pre>
Temps moyen de retour


Démonstration.
   Soient (i, j ) 2 S    S, n 2 N et k = 1, . . . , n. On définit

                        Ek = fXn = j, Ti ,j = k j X0 = i g.



      N. Boussaha ()           Processus Aléatoires (1)    USTHB, 2024-2025   27 / 66
</pre>

</details>
<details>
<summary>PDF page 49</summary>

<pre>
Temps moyen de retour


Démonstration.
   Soient (i, j ) 2 S    S, n 2 N et k = 1, . . . , n. On définit

                        Ek = fXn = j, Ti ,j = k j X0 = i g.
    Il est clair que pour tout k 6= k 0 , on a Ek \ Ek 0 = ∅.



      N. Boussaha ()             Processus Aléatoires (1)     USTHB, 2024-2025   27 / 66
</pre>

</details>
<details>
<summary>PDF page 50</summary>

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
<summary>PDF page 51</summary>

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
<summary>PDF page 52</summary>

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
                                                            < ∞ ) limn !∞ pjj = 0.
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
<summary>PDF page 53</summary>

<pre>
Récurrence nulle et récurrence positive



Definition
  Un état de récurrence j est récurrent positif si µj < ∞.
  Il est dit récurrent nul si µj = ∞.

    Les récurrences positive et nulle sont des propriétés de classe.



     N. Boussaha ()           Processus Aléatoires (1)    USTHB, 2024-2025   30 / 66
</pre>

</details>
<details>
<summary>PDF page 54</summary>

<pre>
Récurrence nulle et récurrence positive



Definition
  Un état de récurrence j est récurrent positif si µj < ∞.
  Il est dit récurrent nul si µj = ∞.

      Les récurrences positive et nulle sont des propriétés de classe.
Démonstration.
TD.



       N. Boussaha ()           Processus Aléatoires (1)   USTHB, 2024-2025   30 / 66
</pre>

</details>
