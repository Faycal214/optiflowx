<details>
<summary>Course slide 45</summary>

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
<details>
<summary>Course slide 46</summary>

<pre>
Distribution stationnaire


Example
S = N, avec une chaîne de naissance pure de taux λi = λ(i+1).

Question : étudier la récurrence et l’existence d’une distribution stationnaire.

      N. Boussaha ()          Processus Aléatoires (1)   USTHB, 2024-2025   46 / 66
</pre>

</details>
<details>
<summary>Course slide 47</summary>

<pre>
Distribution stationnaire

Theorem
Si une chaîne de Markov est irréductible et récurrente positive, alors la
distribution stationnaire est unique.


      N. Boussaha ()          Processus Aléatoires (1)   USTHB, 2024-2025   47 / 66
</pre>

</details>
<details>
<summary>Course slide 48</summary>

<pre>
Distribution stationnaire

Proposition
Pour une chaîne irréductible finie, il existe une unique distribution
stationnaire.


      N. Boussaha ()          Processus Aléatoires (1)   USTHB, 2024-2025   48 / 66
</pre>

</details>
<details>
<summary>Course slide 49</summary>

<pre>
Distribution stationnaire


Propriété
Si la chaîne est irréductible et possède une distribution stationnaire,
elle est récurrente positive.


      N. Boussaha ()          Processus Aléatoires (1)   USTHB, 2024-2025   49 / 66
</pre>

</details>
<details>
<summary>Course slide 50</summary>

<pre>
Distribution stationnaire


Interprétation
La distribution stationnaire donne la proportion de temps passée dans
chaque état à long terme lorsque les conditions de récurrence positive sont
satisfaites.


      N. Boussaha ()          Processus Aléatoires (1)   USTHB, 2024-2025   50 / 66
</pre>

</details>
<details>
<summary>Course slide 51</summary>

<pre>
Distribution limite

Definition
On dit qu’une chaîne admet une distribution limite si, pour tout i, j 2 S,
                                                          (n )
                                       lim p = πj .
                                      n !∞ ij


      N. Boussaha ()          Processus Aléatoires (1)   USTHB, 2024-2025   51 / 66
</pre>

</details>
<details>
<summary>Course slide 52</summary>

<pre>
Distribution limite

Theorem
Si une chaîne de Markov est irréductible, récurrente positive et apériodique,
alors elle est ergodique et la distribution limite existe.


      N. Boussaha ()          Processus Aléatoires (1)   USTHB, 2024-2025   52 / 66
</pre>

</details>
<details>
<summary>Course slide 53</summary>

<pre>
Distribution limite

Theorem
Si (Xn )n 2 N est une chaîne de Markov ergodique alors la distribution
limite existe. De plus
                                    (n )
                       π j = lim pij       = πj ,          8i, j 2 S.
                             n !∞


    Autrement dit, si la distribution limite existe, alors c’est une
distribution stationnaire, i.e., π = π P.

    Il faut préciser ici que limn !∞ Pn = P est une matrice stochastique
    et dont toutes ses lignes sont identiques.



      N. Boussaha ()            Processus Aléatoires (1)                USTHB, 2024-2025   53 / 66
</pre>

</details>
<details>
<summary>Course slide 54</summary>

<pre>
Distribution limite


Généralisation
Si une Chaîne de Markov contient une seule classe de communication
fermée et ergodique C , et que tous les autres états sont transitoires,
alors la la distribution limite existe. avec une masse de probabilité
entièrement concentrée sur C , avec
                    (
                                        (n )
                       π j = limn !∞ pjj = π j , 8j 2 C ,
                       πi = 0                     8i 2
                                                     / C.



     N. Boussaha ()         Processus Aléatoires (1)         2024-2025   54 / 66
</pre>

</details>
<details>
<summary>Course slide 55</summary>

<pre>
Distribution limite



Exercice :
Un nombre X1 est choisi uniformément parmi les entiers 1 à 6 . Supposons
que pour n &gt; 1, Xn est choisi uniformément parmi 1, 2, . . . , Xn 1 .
  1   Montrer que : fXn , n 2 Ng est une CM.
  2   Trouver la matrice de transition.
  3   Calculer limn !∞ Pn .



       N. Boussaha ()            Processus Aléatoires (1)     USTHB, 2024-2025   55 / 66
</pre>

</details>
<details>
<summary>Course slide 56</summary>

<pre>
Distribution limite



Remarque
Si la Chaîne est récurrente périodique de période d 2
                              (
                        (n )     0 Si n 6 0 mod (d ) ,
                   lim pjj =      d
                                     Si n mod (d ) .
                  n !∞           µ  j


    Il est clair dans ce cas, la distribution limite n’existe pas.



     N. Boussaha ()            Processus Aléatoires (1)     USTHB, 2024-2025   56 / 66
</pre>

</details>
<details>
<summary>Course slide 57</summary>

<pre>
Distribution limite


Example
       0 1                  1 1
P=                    ,π=   2, 2   et π .
       1 0



      N. Boussaha ()                Processus Aléatoires (1)   USTHB, 2024-2025   57 / 66
</pre>

</details>
<details>
<summary>Course slide 58</summary>

<pre>
Distribution limite


Example
        0 1                  1 1
P=                     ,π=   2, 2   et π .
        1 0

Remarque
Bien que la distribution stationnaire existe et soit unique, la distribution
limite n’existe pas.



      N. Boussaha ()                Processus Aléatoires (1)   USTHB, 2024-2025   58 / 66
</pre>

</details>
<details>
<summary>Course slide 59</summary>

<pre>
Complément comportement asymptotique


Pour compléter l’étude asymptotique d’une Chaîne de Markov, on examine
                          (n )
le comportement de lim pij en fonction de la nature des états :
                            n !∞

                                                                                             (n )
                        i                                          j                   lim p
                                                                                      n !∞ ij
  transitoire, récurrent (nul ou positif)           transitoire, récurrent nul                0
               récurrent nul                            récurrent positif                     0
                 transitoire                     récurrent positif apériodique                *
                 transitoire                      récurrent positif périodique                *
             récurrent positif              récurrent positif dans une autre classe       0
      récurrent positif apériodique          récurrent positif dans la même classe        πj
       récurrent positif périodique          récurrent positif dans la même classe            *


* mérite une discussion.



       N. Boussaha ()                   Processus Aléatoires (1)           USTHB, 2024-2025       59 / 66
</pre>

</details>
<details>
<summary>Course slide 60</summary>

<pre>
Complément comportement asymptotique
Calcul des probabilités d absorption


      On va se limiter ici aux cas où i est transitoire et j est ergodique.
      Pour cela, on définit ce qu’on appelle les probabilités d’absorption :
      Soit T : ensemble des états transitoires et Ck , k                     1, les classes
      fermés ergodiques.
      La probabilité que la chaîne, partant de i, atteigne la classe ergodique
      Ck pour la 1ère fois en n transitions est donnée par :
                          (n )
                         π i (Ck ) = P(Xn 2 Ck j X0 = i ), 8n 2 N .

      La probabilité d’absorption de la chaîne par Ck partant de i est alors :
                                                          ∞
                                                          [
              π i (Ck ) = ∑ π i (Ck ) = P(
                                   (n )
                                                                 fXn 2 Ck g j X0 = i )      1.
                             n 1                          n =1


        N. Boussaha ()                    Processus Aléatoires (1)            USTHB, 2024-2025   60 / 66
</pre>

</details>
<details>
<summary>Course slide 61</summary>

<pre>
Complément comportement asymptotique
Calcul des probabilités d absorption



Theorem
La probabilité π i (Ck ), pour i 2 T, est (la plus petite) solution (positive)
du système
                       π i (Ck ) = ∑ pij + ∑ pij π j (Ck ).
                                        j 2C            j 2T

      Si T est fini la solution est unique.



        N. Boussaha ()                 Processus Aléatoires (1)   USTHB, 2024-2025   61 / 66
</pre>

</details>
<details>
<summary>Course slide 62</summary>

<pre>
à
Démonstration.
Soit i 2 T. On a :

                                        (Ck ) + ∑ π i (Ck )
                                 (1 )                           (n )
                  π i (Ck ) = π i
                                                     n 2

                                ∑ pij + ∑ ∑ pij πj
                                                                       (n 1 )
                           =                                                    (Ck )
                               j 2C k           n 2 j 2T

                                ∑ pij + ∑ pij ∑ πj
                                                                       (n 1 )
                           =                                                    (Ck )
                               j 2C k           j 2T           n 2

                                ∑ pij + ∑ pij ∑ πj
                                                                       (n )
                           =                                                  (Ck )
                               j 2C k           j 2T           n 1

                           =    ∑ pij + ∑ pij πj (Ck )
                               j 2C k           j 2T



      N. Boussaha ()                Processus Aléatoires (1)                      USTHB, 2024-2025   62 / 66
</pre>

</details>
<details>
<summary>Course slide 63</summary>

<pre>
Complément comportement asymptotique
Utilité du calcul des probabilités d absorption




Exercice 2 (Série1) :
Deux joueurs A et B, disposent d’une fortune initiale de a et b DA
respectivement, où a et b sont des nombres pairs positifs. Ils jouent au jeu
de hasard suivant : la mise est de 2 DA par partie ; les parties sont
indépendantes et à chacune d’entre elle le joueur A a une probabilité p de
gagner, avec 0 &lt; p &lt; 1. Le jeu se termine dès que l’un des joueurs est
ruiné.
    Trouver la probabilité pour que le joueur N 1 gagne la partie.




        N. Boussaha ()                Processus Aléatoires (1)   USTHB, 2024-2025   63 / 66
</pre>

</details>
<details>
<summary>Course slide 64</summary>

<pre>
Complément comportement asymptotique
Utilité du calcul des probabilités d absorption




Theorem (Revenant à notre objectif :)
Si i est transitoire et j est ergodique :
                          (n )             (n )
                     lim p = π i (C ) lim pjj = π i (C ) lim π j .
                    n !∞ ij          n !∞               n !∞

avec C est la classe de communication contenant j.



        N. Boussaha ()                Processus Aléatoires (1)   USTHB, 2024-2025   64 / 66
</pre>

</details>
<details>
<summary>Course slide 65</summary>

<pre>
Complément comportement asymptotique
Calcul des probabilités d absorption




Exercice (Série1) :
Calculer limn !∞ P(n )
                         0                      1
                       0.5 0 0.2 0 0.3 0     0
                     B 0 0.5 0.3 0 0.2 0     0 C
                     B                          C
                     B 0   0 0.3 0.7 0   0   0  C
                     B                          C
                     B
                   P=B 0   0 0.3 0.7 0   0   0 CC.
                     B 0   0  0   0  0  0.2 0.8 C
                     B                          C
                     @ 0   0  0   0 0.2 0.8 0.2 A
                        0  0  0   0 0.8 0 0.2




        N. Boussaha ()                 Processus Aléatoires (1)   USTHB, 2024-2025   65 / 66
</pre>

</details>
<details>
<summary>Course slide 66</summary>

<pre>
Bibliographie



   KERNANE. T. Processus stochastiques. Polycopié du cours.
   L3-Probabilités et Statistique. USTHB
   Rabehi. N. Processus stochastiques. Polycopié du cours. M1-MSPRO.
   USTHB. 2023-2024.
   Mazliak, L., Priouret, P., &amp; Baldi, P. Martingales et chaînes de
   Markov. 1998.




     N. Boussaha ()          Processus Aléatoires (1)   USTHB, 2024-2025   66 / 66
</pre>

</details>
