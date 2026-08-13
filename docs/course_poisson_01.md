<details>
<summary>Course slide 1</summary>

<pre>
                       Processus Aléatoires

 Master Modélisation Stochastique et Prévision en Recherche
                 Opérationnelle (MSPRO)
                 Chapitre 2 : Processus de Poisson (PP)

                   Laboratoire RECITS (Equipe STEP)
                               N. Boussaha


                          USTHB, 2024-2025




N. Boussaha ()               Processus Aléatoires (2)   USTHB, 2024-2025   1 / 22
</pre>

</details>
<details>
<summary>Course slide 2</summary>

<pre>
Introduction

    Un Processus de Poisson (PP), nommé d’après le mathématicien
    français Siméon Denis Poisson et la loi du même nom, est un
    processus à temps continu et à valeurs entières positives (cas
    particulier d’une Chaîne de Markov à Temps Continu CMTC).
    Sa grande popularité dans les applications vient notamment du fait
    que beaucoup de calculs le concernant sont explicites.
    On se propose d’étudier la répartition dans le temps d’instants
    aléatoires ou se produisent certains évènements spécifiques, comme
    par exempls :
       les appels dans un central téléphonique,
       les émissions de particules radioactives,
       nombres de pannes d’une machines
       les arrivés de clients devant un guichet...

     N. Boussaha ()            Processus Aléatoires (2)   USTHB, 2024-2025   2 / 22
</pre>

</details>
<details>
<summary>Course slide 3</summary>

<pre>
Définition d’un processus de comptage



Definition
Soit la v .a. N (t ) associée aux nombres d’occurrences se produisant dans
intervalle ]0, t ] . Un processus (N (t ))t 0 est dit processus de
dénombrement ou processus de comptage si :
    8t      0,     N (t ) 2 N.
    Si s &lt; t ) N (s )       N (t ).
    Si s &lt; t, alors N (t ) N (s ) représente le nombre d’occurrences
    (réalisations) d’événements sur l’intervalle ]s, t ].




      N. Boussaha ()              Processus Aléatoires (2)   USTHB, 2024-2025   3 / 22
</pre>

</details>
<details>
<summary>Course slide 4</summary>

<pre>
Définition d’un processus de poisson (PP)
Definition (1)
Un processus de dénombrement (N (t ))t 0 est dit processus de Poisson
de taux λ (λ &gt; 0) si :
    N (0) = 0, p.s.
    Le processus (N (t ))t 0 est à accroissements indépendants, i.e.,
    pour toute suite t1 &lt; t2 &lt;        &lt; tn , les v .a. N (ti +1 ) N (ti ) et
    N (tj +1 ) N (tj ), avec i 6= j, sont indépendantes.
    Le processus (N (t ))t 0 est à accroissements stationnaires, i.e.,
    pour toute suite t1 &lt; t2 &lt;      &lt; tn , les v .a. N (ti + s ) N (ti ) et
    N (tj + s ) N (tj ) ont même distribution (N (t ) N (s ) a la même
    distribution que N (t s ) si s &lt; t).
                                                                  o (h )
    8h &gt; 0,       P(N (h) = 1) = λh + o (h) avec lim                h = 0.
                                                           h !0

    8h &gt; 0,       P(N (h )   2) = o (h ).
     N. Boussaha ()             Processus Aléatoires (2)             USTHB, 2024-2025   4 / 22
</pre>

</details>
<details>
<summary>Course slide 5</summary>

<pre>
Définition d’un processus de poisson (PP)


Definition (2)
Un processus de dénombrement (N (t ))t 0 est dit processus de Poisson
de taux λ (λ &gt; 0) si :
    N (0) = 0, p.s.
    Le processus (N (t ))t 0 est à accroissements indépendants,
    Le nombre d’occurences dans intervalle de langeur t suit une loi de
    poisson de paramètre λt,
                                                               (λt )n
    i.e., 8n 2 N 8t 0 : P(N (t + s ) N (s ) = n) = e λt n! ,
    (indépendante de s).
    Donc par construction un PP est un processus à accroissements
    stattionnaires.


     N. Boussaha ()          Processus Aléatoires (2)   USTHB, 2024-2025   5 / 22
</pre>

</details>
<details>
<summary>Course slide 6</summary>

<pre>
Équivalence des deux définitions
     Montrons que les deux définitions sont équivalentes.

Démonstration.
[1 =) 2] Pour tout n 2 N, posons pn (t ) = P(N (t ) = n). On a alors :

P (N (t + h ) = 0)     =   p0 (t + h) = P (0 occurrence dans ]0, t + h])
                       =   P (0 occurrence dans ]0, t ] et 0 occurrence dans ]t, t + h])
                       =   P (0 occurrence dans ]0, t ]) P (0 occurrence dans ]t, t +
                       =   P(N (t ) = 0) (1 λh + o (h))
                       =   p0 (t )(1 λh + o (h))


                   =) p0 (t + h) p0 (t ) = p0 (t )( λh + o (h))
                      p0 (t + h) p0 (t )                o (h )
                   =)                     = λp0 (t ) +         p0 (t )
                                h                         h
                   =) p00 (t ) = λp0 (t )
                   =) p0 (t ) = e λt .
      N. Boussaha ()               Processus Aléatoires (2)     USTHB, 2024-2025   6 / 22
</pre>

</details>
<details>
<summary>Course slide 7</summary>

<pre>
Démonstration.
D’autre part, on a :

           P (N (t + h ) = n )
     =     pn (t + h) = P (n occurrence dans ]0, t + h])
     =     P (fn occurrence dans ]0, t ] et 0 occurrence dans ]t, t + h]g ou
           fn 1 occurrence dans ]0, t ] et 0 occurrence dans ]t, t + h]g)
     =     P (n occurrence dans ]0, t ]) P (0 occurrence dans ]t, t + h])
           +P (n 1 occurrence dans ]0, t ]) P (1 occurrence dans ]t, t + h])
     =     pn (t )(1 λh + o (h)) + pn 1 (t )(λh + o (h))

        =) pn (t +hh) pn (t ) = pn (t )    o (h )
                                        λ+ h      + pn 1 (t ) λ + o (hh )
        =) pn0 (t ) = λpn (t ) + λpn 1 (t ).
! Équation différentielle linéaire du premier ordre non homogène.
! Résolution par la méthode de la variation de la constante,
  ou par la méthode de superposition (somme de la solution générale et la solution
  particulière).
        N. Boussaha ()             Processus Aléatoires (2)        USTHB, 2024-2025   7 / 22
</pre>

</details>
<details>
<summary>Course slide 8</summary>

<pre>
Équivalence des deux définitions

Démonstration.
Suite de la preuve
    Pour n = 1
     On pose alors :              p1 (t ) = C 1 (t )e λt avec p1 (0) = 0.
     En dérivant, on obtient :    p10 (t ) = C10 (t ) e λt           λC1 (t )e λt .
                                   p10 (t ) = λp1 (t ) + λp0 (t )
     En mettant en parallèle :
                                            = λC1 (t )e λt + λe λt
     En identifiant, on a :          0
                                  C1 (t ) = λ.
     En intégrant, on obtient : C1 (t ) = λt.
                                                            (λt )1
     Finalement,                  pn (t ) = e λt              1! .
    Pour n = 2...



     N. Boussaha ()              Processus Aléatoires (2)              USTHB, 2024-2025   8 / 22
</pre>

</details>
<details>
<summary>Course slide 9</summary>

<pre>
Démonstration.
Suite de la preuve
                                                                                                (λt )n
    On montre par récurrence sur n que :                              pn (t ) = e λt              n! .
                                                                                                       n 1
    Supposons qu’elle est vraie pour n          1, i.e., :           pn 1 (t ) = e             λt λt )
                                                                                                 (
                                                                                                           .
                                                                                                  (n 1 ) !
    D’après l’équation différentielle obtenue précédemment :
                                                                         n 1
                                                             λt ( λt )
                        pn0 (t ) =    λpn (t ) + λe                             .
                                                                (n       1) !

     On pose alors :                 pn (t ) = C n (t )e λt avec pn (0) = 0 8 n                            1.
     En dérivant, on obtient :       pn0 (t ) = Cn0 (t ) e λt              λCn (t )e          λt .

                                                   (λt )n 1
     En identifiant, on a :           Cn0 (t ) = λ (n 1 )! .
                                                (λt )n
     En intégrant, on obtient :      Cn (t ) = n! .
                                                       (λt )n
     Finalement,                     pn (t ) = e λt n! .


     N. Boussaha ()               Processus Aléatoires (2)                          USTHB, 2024-2025     9 / 22
</pre>

</details>
<details>
<summary>Course slide 10</summary>

<pre>
Équivalence des deux définitions




Démonstration.
[2 =) 1] TD.




     N. Boussaha ()    Processus Aléatoires (2)   USTHB, 2024-2025   10 / 22
</pre>

</details>
<details>
<summary>Course slide 11</summary>

<pre>
Propriété




Exemple
Le nombre d’appels reçus par un centre de service client suit un processus
de Poisson ( (N (t ))t 0 avec un taux de λ = 2 appels par minute.
   Sachant que 4 appels ont été reçus en 3 minutes, quelle est la
probabilité que 3 d’entre eux aient été reçus dans les 2 premières minutes ?




      N. Boussaha ()          Processus Aléatoires (2)   USTHB, 2024-2025   11 / 22
</pre>

</details>
