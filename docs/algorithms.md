# Algorithms in OptiFlowX

## Genetic Algorithm (GA)

In computer science and operations research, a **genetic algorithm (GA)** is a metaheuristic inspired by the process of natural selection that belongs to the larger class of evolutionary algorithms (EA). Genetic algorithms are commonly used to generate high-quality solutions to optimization and search problems via biologically inspired operators such as selection, crossover, and mutation. Some examples of GA applications include optimizing decision trees for better performance, solving sudoku puzzles, hyperparameter optimization, and causal inference.

### Optimization problems

In a genetic algorithm, a [population](https://en.wikipedia.org/wiki/Population) of [candidate solutions](https://en.wikipedia.org/wiki/Candidate_solution) (called individuals, creatures, organisms, or phenotypes) to an optimization problem is evolved toward better solutions. Each candidate solution has a set of properties (its [chromosomes](https://en.wikipedia.org/wiki/Chromosome) or [genotype](https://en.wikipedia.org/wiki/Genotype)) which can be mutated and altered; traditionally, solutions are represented in binary as strings of 0s and 1s, but other encodings are also possible.

The evolution usually starts from a population of randomly generated individuals, and is an iterative process, with the population in each iteration called a generation. In each generation, the [fitness](https://en.wikipedia.org/wiki/Fitness_(biology)) of every individual in the population is evaluated; the fitness is usually the value of the [objective function](https://en.wikipedia.org/wiki/Objective_function) in the optimization problem being solved. The more fit individuals are stochastically selected from the current population, and each individual's genome is modified (recombined and possibly randomly mutated) to form a new generation. The new generation of candidate solutions is then used in the next iteration of the algorithm. Commonly, the algorithm terminates when either a maximum number of generations has been produced, or a satisfactory fitness level has been reached for the population.

A typical genetic algorithm requires:

* A [genetic representation](https://en.wikipedia.org/wiki/Genetic_representation) of the solution domain,
* A [fitness function](https://en.wikipedia.org/wiki/Fitness_function) to evaluate the solution domain.

A standard representation of each candidate solution is as an array of bits (also called bit set or bit string). Arrays of other types and structures can be used in essentially the same way. The main property that makes these genetic representations convenient is that their parts are easily aligned due to their fixed size, which facilitates simple [crossover](https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)) operations. Variable length representations may also be used, but crossover implementation is more complex in this case. Tree-like representations are explored in genetic programming and graph-form representations are explored in evolutionary programming; a mix of both linear chromosomes and trees is explored in gene expression programming.

Once the genetic representation and the fitness function are defined, a GA proceeds to initialize a population of solutions and then to improve it through repetitive application of the mutation, crossover, inversion and selection operators.

### Initialization

The population size depends on the nature of the problem, but typically contains hundreds or thousands of possible solutions. Often, the initial population is generated randomly, allowing the entire range of possible solutions (the [search space](https://en.wikipedia.org/wiki/Feasible_region)). Occasionally, the solutions may be "seeded" in areas where optimal solutions are likely to be found or the distribution of the sampling probability tuned to focus in those areas of greater interest.

### Selection

Main article: [Selection (genetic algorithm)](https://en.wikipedia.org/wiki/Selection_(genetic_algorithm))

During each successive generation, a portion of the existing population is selected to reproduce for a new generation. Individual solutions are selected through a fitness-based process, where fitter solutions (as measured by a fitness function) are typically more likely to be selected. Certain selection methods rate the fitness of each solution and preferentially select the best solutions. Other methods rate only a random sample of the population, as the former process may be very time-consuming.

The fitness function is defined over the genetic representation and measures the quality of the represented solution. The fitness function is always problem-dependent. For instance, in the knapsack problem one wants to maximize the total value of objects that can be put in a knapsack of some fixed capacity. A representation of a solution might be an array of bits, where each bit represents a different object, and the value of the bit (0 or 1) represents whether or not the object is in the knapsack. Not every such representation is valid, as the size of objects may exceed the capacity of the knapsack. The fitness of the solution is the sum of values of all objects in the knapsack if the representation is valid, or 0 otherwise.

### Genetic operators

Main articles: [Crossover (genetic algorithm)](https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)) and [Mutation (genetic algorithm)](https://en.wikipedia.org/wiki/Mutation_(genetic_algorithm)).

The next step is to generate a second generation population of solutions from those selected, through a combination of genetic operators: crossover (also called recombination), and mutation.

For each new solution to be produced, a pair of "parent" solutions is selected for breeding from the pool selected previously. By producing a "child" solution using the above methods of crossover and mutation, a new solution is created which typically shares many of the characteristics of its "parents". New parents are selected for each new child, and the process continues until a new population of solutions of appropriate size is generated. Although reproduction methods that are based on the use of two parents are more "biology inspired", some research suggests that more than two "parents" generate higher quality chromosomes.

These processes ultimately result in the next generation population of chromosomes that is different from the initial generation. Generally, the average fitness will have increased by this procedure for the population, since only the best organisms from the first generation are selected for breeding, along with a small proportion of less fit solutions. These less fit solutions ensure genetic diversity within the genetic pool of the parents and therefore ensure the genetic diversity of the subsequent generation of children.

It is worth tuning parameters such as the mutation probability, crossover probability and population size to find reasonable settings for the problem's complexity class being worked on. A very small mutation rate may lead to [genetic drift](https://en.wikipedia.org/wiki/Genetic_drift) (which is non-ergodic in nature). A recombination rate that is too high may lead to premature convergence of the genetic algorithm. A mutation rate that is too high may lead to loss of good solutions, unless elitist selection is employed. An adequate population size ensures sufficient genetic diversity for the problem at hand, but can lead to a waste of computational resources if set to a value larger than required.

### Termination

This generational process is repeated until a termination condition has been reached. Common terminating conditions are:

* A solution is found that satisfies minimum criteria
* Fixed number of generations reached
* Allocated budget (computation time/money) reached
* The highest ranking solution's fitness is reaching or has reached a plateau such that successive iterations no longer produce better results
* Manual inspection
* Combinations of the above

### Pseudo Code

```text
Initialize population P with N random candidate solutions
Evaluate fitness f(x) for each x in P
For generation = 1 to G:
    Select parent solutions from P (higher-fitness more likely)
    Apply crossover (with probability p_c) to parents to produce offspring
    Apply mutation (with probability p_m) to offspring
    Evaluate fitness of all offspring
    Form new population P by (for example) selecting the top N candidates from parents ∪ offspring
Output the best solution(s) found (highest fitness)
```

---

## Particle Swarm Optimization (PSO)

In [computational science](https://en.wikipedia.org/wiki/Computational_science), **particle swarm optimization (PSO)** is a computational method that optimizes a problem by iteratively trying to improve a candidate solution with regard to a given measure of quality. It solves a problem by having a population of candidate solutions, here dubbed [particles](https://en.wikipedia.org/wiki/Point_particle), and moving these particles around in the search-space according to simple mathematical formulae over the particle's [position](https://en.wikipedia.org/wiki/Position_(vector)) and [velocity](https://en.wikipedia.org/wiki/Velocity). Each particle's movement is influenced by its local best known position, but is also guided toward the best known positions in the search-space, which are updated as better positions are found by other particles. This is expected to move the swarm toward the best solutions.

<img src="https://upload.wikimedia.org/wikipedia/commons/e/ec/ParticleSwarmArrowsAnimation.gif" alt="drawing" width="500"/>

PSO is originally attributed to Kennedy, Eberhart and Shi and was first intended for simulating social behaviour, as a stylized representation of the movement of organisms in a bird flock or fish school. The algorithm was simplified and it was observed to be performing optimization. The book by Kennedy and Eberhart describes many philosophical aspects of PSO and swarm intelligence. An extensive survey of PSO applications is made by Poli. In 2017, a comprehensive review on theoretical and experimental works on PSO has been published by Bonyadi and Michalewicz.

PSO is a metaheuristic as it makes few or no assumptions about the problem being optimized and can search very large spaces of candidate solutions. Also, PSO does not use the [gradient](https://en.wikipedia.org/wiki/Gradient) of the problem being optimized, which means PSO does not require that the optimization problem be [differentiable](https://en.wikipedia.org/wiki/Differentiable_function) as is required by classic optimization methods such as gradient descent and quasi-newton methods. However, metaheuristics such as PSO do not guarantee an optimal solution is ever found.

### Algorithm

A basic variant of the PSO algorithm works by having a population (called a swarm) of candidate solutions (called particles). These particles are moved around in the search-space according to a few simple formulae. The movements of the particles are guided by their own best-known position in the search-space as well as the entire swarm's best-known position. When improved positions are being discovered these will then come to guide the movements of the swarm. The process is repeated and by doing so it is hoped, but not guaranteed, that a satisfactory solution will eventually be discovered.

Formally, let $f: \mathbb{R}^n \rightarrow \mathbb{R}$ be the cost function which must be minimized. The function takes a candidate solution as an argument in the form of a vector of real numbers and produces a real number as output which indicates the objective function value of the given candidate solution. The gradient of $f$ is not known. The goal is to find a solution $a$ for which $f(a) \leq f(b)$ for all $b$ in the search-space, which would mean $a$ is the global minimum.

Let $S$ be the number of particles in the swarm, each having a position $x_i \in \mathbb{R}^n$ in the search-space and a velocity $v_i \in \mathbb{R}^n$. Let $p_i$ be the best known position of particle $i$ and let $g$ be the best known position of the entire swarm. A basic PSO algorithm to minimize the cost function is then

```text
for each particle i = 1, ..., S do
    Initialize the particle's position with a uniformly distributed random vector: xi ~ U(blo, bup)
    Initialize the particle's best known position to its initial position: pi ← xi
    if f(pi) < f(g) then
        update the swarm's best known position: g ← pi
    Initialize the particle's velocity: vi ~ U(-|bup-blo|, |bup-blo|)
while a termination criterion is not met do:
    for each particle i = 1, ..., S do
        for each dimension d = 1, ..., n do
            Pick random numbers: rp, rg ~ U(0,1)
            Update the particle's velocity: vi,d ← w vi,d + φp rp (pi,d-xi,d) + φg rg (gd-xi,d)
        Update the particle's position: xi ← xi + vi
        if f(xi) < f(pi) then
            Update the particle's best known position: pi ← xi
            if f(pi) < f(g) then
                Update the swarm's best known position: g ← pi
```

The values $b_{lo}$ and $b_{up}$ represent the lower and upper boundaries of the search-space respectively. The $w$ parameter is the inertia weight. The parameters $\phi_p$ and $\phi_g$ are often called cognitive coefficient and social coefficient.

The termination criterion can be the number of iterations performed, or a solution where the adequate objective function value is found. The parameters $w$, $\phi_p$, and $\phi_g$ are selected by the practitioner and control the behaviour and efficacy of the PSO method (below).

### Parameter selection

The choice of PSO parameters can have a large impact on optimization performance. Selecting PSO parameters that yield good performance has therefore been the subject of much research.

To prevent divergence ("explosion") the inertia weight must be smaller than 1. The two other parameters can be then derived thanks to the constriction approach, or freely selected, but the analyses suggest convergence domains to constrain them. Typical values are in $[1, 3]$.

The PSO parameters can also be tuned by using another overlaying optimizer, a concept known as meta-optimization, or even fine-tuned during the optimization, e.g., by means of fuzzy logic.

Parameters have also been tuned for various optimization scenarios.

### Neighbourhoods and topologies

The topology of the swarm defines the subset of particles with which each particle can exchange information. The basic version of the algorithm uses the global topology as the swarm communication structure. This topology allows all particles to communicate with all the other particles, thus the whole swarm share the same best position g from a single particle. However, this approach might lead the swarm to be trapped into a local minimum, thus different topologies have been used to control the flow of information among particles. For instance, in local topologies, particles only share information with a subset of particles. This subset can be a geometrical one – for example "the m nearest particles" – or, more often, a social one, i.e. a set of particles that is not depending on any distance. In such cases, the PSO variant is said to be local best (vs global best for the basic PSO).

A commonly used swarm topology is the ring, in which each particle has just two neighbours, but there are many others. The topology is not necessarily static. In fact, since the topology is related to the diversity of communication of the particles, some efforts have been done to create adaptive topologies (SPSO, APSO, stochastic star, TRIBES, Cyber Swarm, and C-PSO)

By using the ring topology, PSO can attain generation-level parallelism, significantly enhancing the evolutionary speed.

### Convergence

In relation to PSO the word convergence typically refers to two different definitions:

* Convergence of the sequence of solutions (aka, stability analysis, converging) in which all particles have converged to a point in the search-space, which may or may not be the optimum,
* Convergence to a local optimum where all personal bests p or, alternatively, the swarm's best known position g, approaches a local optimum of the problem, regardless of how the swarm behaves.

Convergence of the sequence of solutions has been investigated for PSO. These analyses have resulted in guidelines for selecting PSO parameters that are believed to cause convergence to a point and prevent divergence of the swarm's particles (particles do not move unboundedly and will converge to somewhere). However, the analyses were criticized by Pedersen for being oversimplified as they assume the swarm has only one particle, that it does not use stochastic variables and that the points of attraction, that is, the particle's best known position p and the swarm's best known position g, remain constant throughout the optimization process. However, it was shown that these simplifications do not affect the boundaries found by these studies for parameter where the swarm is convergent. Considerable effort has been made in recent years to weaken the modeling assumption utilized during the stability analysis of PSO, with the most recent generalized result applying to numerous PSO variants and utilized what was shown to be the minimal necessary modeling assumptions.

Convergence to a local optimum has been analyzed for PSO and It has been proven that PSO needs some modification to guarantee finding a local optimum.

This means that determining the convergence capabilities of different PSO algorithms and parameters still depends on empirical results. One attempt at addressing this issue is the development of an "orthogonal learning" strategy for an improved use of the information already existing in the relationship between p and g, so as to form a leading converging exemplar and to be effective with any PSO topology. The aims are to improve the performance of PSO overall, including faster global convergence, higher solution quality, and stronger robustness. However, such studies do not provide theoretical evidence to actually prove their claims.

---

## Simulated Annealing (SA)

**Simulated annealing (SA)** is a [probabilistic technique](https://en.wikipedia.org/wiki/Probabilistic_algorithm) for approximating the [global optimum](https://en.wikipedia.org/wiki/Global_optimum) of a given function. Specifically, it is a [metaheuristic](https://en.wikipedia.org/wiki/Metaheuristic) to approximate [global optimization](https://en.wikipedia.org/wiki/Global_optimization) in a large search space for an optimization problem. For large numbers of local optima, SA can find the global optimum. It is often used when the search space is discrete (for example the traveling salesman problem, the boolean satisfiability problem, protein structure prediction, and job-shop scheduling). For problems where a fixed amount of computing resource is available, finding an approximate global optimum may be more relevant than attempting to find a precise local optimum. In such cases, SA may be preferable to exact algorithms such as gradient descent or branch and bound. The problems solved by SA are currently formulated by an objective function of many variables, subject to several mathematical constraints. In practice, a constraint violation can be penalized as part of the objective function.

<img src="https://upload.wikimedia.org/wikipedia/commons/1/10/Travelling_salesman_problem_solved_with_simulated_annealing.gif" alt="drawing" width="350"/>

The name of the algorithm comes from [annealing in metallurgy](https://en.wikipedia.org/wiki/Annealing_(metallurgy)), a technique involving heating and controlled cooling of a material to alter its physical properties. This notion of slow cooling implemented in the simulated annealing algorithm is interpreted as a slow decrease in the probability of accepting worse solutions as the solution space is explored. Accepting worse solutions allows for a more extensive search for the global optimal solution. Simulated annealing algorithms work by progressively decreasing the temperature from an initial positive value to zero. At each time step, the algorithm randomly selects a solution close to the current one, measures its quality, and moves to it according to the temperature-dependent probabilities of selecting better or worse solutions.

### Overview

The state s of some physical systems, and the function E(s) to be minimized, is analogous to the internal energy of the system in that state. The goal is to bring the system, from an arbitrary initial state, to a state with the minimum possible energy.

<img src="https://upload.wikimedia.org/wikipedia/commons/d/d5/Hill_Climbing_with_Simulated_Annealing.gif" alt="drawing" width="650"/>

### The basic iteration

At each step, the simulated annealing heuristic considers some neighboring state s* of the current state s, and probabilistically decides between moving the system to state s* or staying in state s. These probabilities ultimately lead the system to move to states of lower energy. Typically this step is repeated until the system reaches a state that is good enough for the application, or until a given computation budget has been exhausted.

### The neighbors of a state

Optimization of a solution involves evaluating the neighbors of a state of the problem, which are new states produced through conservatively altering a given state. For example, in the traveling salesman problem each state is typically defined as a permutation of the cities to be visited, and the neighbors of any state are the set of permutations produced by swapping any two of these cities. The well-defined way in which the states are altered to produce neighboring states is called a "move", and different moves give different sets of neighboring states. These moves usually result in minimal alterations of the last state, in an attempt to progressively improve the solution through iteratively improving its parts (such as the city connections in the traveling salesman problem). It is even better to reverse the order of an interval of cities. This is a smaller move since swapping two cities can be achieved by twice reversing an interval.

Simple [heuristics](https://en.wikipedia.org/wiki/Heuristic) like [hill climbing](https://en.wikipedia.org/wiki/Hill_climbing), which move by finding better neighbor after better neighbor and stop when they have reached a solution which has no neighbors that are better solutions, cannot guarantee to lead to any of the existing better solutions – their outcome may easily be just a local optimum, while the actual best solution would be a global optimum that could be different. Metaheuristics use the neighbors of a solution as a way to explore the solution space, and although they prefer better neighbors, they also accept worse neighbors in order to avoid getting stuck in local optima; they can find the global optimum if run for a long enough amount of time.

### Acceptance probabilities

The probability of making the transition from the current state $\textbf{s}$ to a candidate new state $\textbf{s}_{\text{new}}$ is specified by an acceptance probability function $\textbf{P}(e, e_{\text{new}}, T)$, that depends on the energies $e = E(\textbf{s})$ and $e_{\text{new}} = E(\textbf{s}_{\text{new}})$ of the two states, and on a global time-varying parameter $T$ called the temperature. States with a smaller energy are better than those with a greater energy. The probability function $\textbf{P}$ must be positive even when $e_{\text{new}}$ is greater than $e$. This feature prevents the method from becoming stuck at a local minimum that is worse than the global one.

When $T$ tends to zero, the probability $\textbf{P}(e, e_{\text{new}}, T)$ must tend to zero if $e_{\text{new}} > e$ and to a positive value otherwise. For sufficiently small values of $T$, the system will then increasingly favor moves that go "downhill" (i.e., to lower energy values), and avoid those that go "uphill." With $T = 0$ the procedure reduces to the [greedy algorithm](https://en.wikipedia.org/wiki/Greedy_algorithm), which makes only the downhill transitions.

In the original description of simulated annealing, the probability $\textbf{P}(e, e_{\text{new}}, T)$ was equal to 1 when $e_{\text{new}} < e$, i.e., the procedure always moved downhill when it found a way to do so, irrespective of the temperature. Many descriptions and implementations of simulated annealing still take this condition as part of the method's definition. However, this condition is not essential for the method to work.

The $\textbf{P}$ function is usually chosen so that the probability of accepting a move decreases when the difference $e_{\text{new}} - e$ increases, that is, small uphill moves are more likely than large ones. However, this requirement is not strictly necessary, provided that the above requirements are met.

Given these properties, the temperature $T$ plays a crucial role in controlling the evolution of the state $\textbf{s}$ of the system with regard to its sensitivity to the variations of system energies. To be precise, for a large $T$, the evolution of $\textbf{s}$ is sensitive to coarser energy variations, while it is sensitive to finer energy variations when $T$ is small.

### The annealing schedule

The name and inspiration of the algorithm demand an interesting feature related to the temperature variation to be embedded in the operational characteristics of the algorithm. This necessitates a gradual reduction of the temperature as the simulation proceeds. The algorithm starts initially with $T$ set to a high value (or infinity), and then it is decreased at each step following some annealing schedule—which may be specified by the user but must end with $T = 0$ towards the end of the allotted time budget. In this way, the system is expected to wander initially towards a broad region of the search space containing good solutions, ignoring small features of the energy function; then drift towards low-energy regions that become narrower and narrower, and finally move downhill according to the [steepest descent](https://en.wikipedia.org/wiki/Steepest_descent) heuristic.

For any given finite problem, the probability that the simulated annealing algorithm terminates with a global optimal solution approaches 1 as the annealing schedule is extended. This theoretical result, however, is not particularly helpful, since the time required to ensure a significant probability of success will usually exceed the time required for a complete search of the solution space.

### Pseudocode

```text
Let s = s0 (initial solution), T0 = initial temperature
For k = 1 to k_max:
    T = temperature_schedule(T0, k)  // e.g. linear or exponential cooling
    s_new = random_neighbour(s)      // small random change
    Compute E_new = cost(s_new), E = cost(s)
    If E_new < E:
        s = s_new
    Else if exp[-(E_new - E) / T] ≥ random(0,1):
        s = s_new
Return the best solution s found
```

---

## Bayesian Optimization

**Bayesian optimization** is a sequential design strategy for global optimization of black-box functions, that does not assume any functional forms. It is usually employed to optimize expensive-to-evaluate functions. With the rise of artificial intelligence innovation in the 21st century, Bayesian optimization algorithms have found prominent use in machine learning problems for optimizing hyperparameter values.

### Strategy

Bayesian optimization is used on problems of the form $\max_{x \in X} f(x)$, with $X$ being the set of all possible parameters $x$, typically with less than or equal to 20 dimensions for optimal usage ($X \rightarrow \mathbb{R}^d \mid d \leq 20$), and whose membership can easily be evaluated. Bayesian optimization is particularly advantageous for problems where $f(x)$ is difficult to evaluate due to its computational cost. The objective function, $f$, is continuous and takes the form of some unknown structure, referred to as a "black box". Upon its evaluation, only $f(x)$ is observed and its derivatives are not evaluated.

Since the objective function is unknown, the Bayesian strategy is to treat it as a random function and place a prior over it. The prior captures beliefs about the behavior of the function. After gathering the function evaluations, which are treated as data, the prior is updated to form the posterior distribution over the objective function. The posterior distribution, in turn, is used to construct an acquisition function (often also referred to as infill sampling criteria) that determines the next query point.

There are several methods used to define the prior/posterior distribution over the objective function. The most common two methods use Gaussian processes in a method called kriging. Another less expensive method uses the Parzen-Tree Estimator to construct two distributions for 'high' and 'low' points, and then finds the location that maximizes the expected improvement.

Standard Bayesian optimization relies upon each $x \in X$ being easy to evaluate, and problems that deviate from this assumption are known as exotic Bayesian optimization problems. Optimization problems can become exotic if it is known that there is noise, the evaluations are being done in parallel, the quality of evaluations relies upon a tradeoff between difficulty and accuracy, the presence of random environmental conditions, or if the evaluation involves derivatives.

### Acquisition function

Examples of acquisition functions include :

* probability of improvement
* expected improvement
* Bayesian expected losses
* Upper Confidence Bound (UCB) or lower confidence bounds
* Thompson sampling

and hybrids of these. They all trade-off exploration and exploitation so as to minimize the number of function queries. As such, Bayesian optimization is well suited for functions that are expensive to evaluate.

### Workflow

1. **Initialization**: Sample an initial set of hyperparameters (e.g. randomly or space-filling) and evaluate the objective (training/validation) at these points.

2. **Model update**: Fit a Gaussian Process (GP) to all observed data $\{(x_i, f(x_i))\}$. The GP yields a posterior mean $\mu(x)$ and variance $\sigma^2(x)$ for the objective at any $x$.

3. **Acquisition maximization**: Define an acquisition function $a(x)$ (e.g. Expected Improvement (EI) or Probability of Improvement (PI)). The acquisition function uses the GP’s posterior to score potential hyperparameters. Find $x_{\text{next}} = \arg\max_x a(x)$ (often via an inner optimization or sampling).

4. **Evaluate**: Compute the true objective $f(x_{\text{next}})$ (e.g. model validation score with these hyperparameters) and add to observations.

5. **Iterate**: Go back to Model update (step 2) and repeat until budget is exhausted.

6. **Return**: the best hyperparameters seen (or those with highest posterior mean).

### Key Mathematical Steps

* **Gaussian Process (GP)**: a prior over functions. Given $n$ observed points $\{x_i, y_i\}$, the GP computes a posterior mean $\mu_n(x)$ and variance $\sigma_n^2(x)$ at any $x$. These use a covariance kernel $k(x, x')$ and prior mean (often zero).

* **Acquisition Function**: quantifies utility of sampling $x$. For example, Expected Improvement at $x$ is

    $EI(x) = E[\max(f(x) - f_{\text{best}}, 0)] = (\mu_n(x) - f_{\text{best}})\Phi(z) + \sigma_n(x)\phi(z)$,

   where $z = [\mu_n(x) - f_{\text{best}}] / \sigma_n(x)$, and $\Phi, \phi$ are the CDF/pdf of the normal distribution. This encourages high $\mu_n(x)$ and high $\sigma_n(x)$.

* **Optimization**: At each iteration the acquisition $a(x)$ is maximized (often by separate optimization or grid) to pick next sample.

* **Updating**: After sampling, the GP posterior is updated with the new point.

### Parameters

* **Surrogate parameters**: kernel type (RBF, Matern, etc.), noise level.

* **Acquisition type**: EI, PI, or Upper Confidence Bound (UCB).

* **Initial design size**: number of points to sample before BO loop (often random).

* **Budget $N$**: total number of evaluations (objective runs).

### Pseudocode

```text
Place a Gaussian process prior on f
Observe f at n0 initial points (e.g. random or Latin hypercube design); n = n0
While n ≤ N:
    Fit GP to all observed data {(x_i, f(x_i))}
    Compute acquisition function a(x) from the GP posterior
    x_n = argmax_x a(x)          // choose next hyperparameters by maximizing acquisition
    Observe y_n = f(x_n)         // evaluate the objective
    Add (x_n, y_n) to data
    n = n + 1
Return the best x found (maximizing f or μ)
```

---

## Tree-structured Parzen Estimator (TPE)

**The Tree-structured Parzen Estimator (TPE)** is a sequential model-based optimization (SMBO) algorithm, a subfield of Bayesian optimization, designed for the efficient tuning of hyperparameters. It is particularly effective for optimizing functions that are computationally expensive to evaluate, a common challenge in machine learning. Instead of modeling the performance of hyperparameters directly, TPE models the probability of observing hyperparameters given a certain performance score, using Kernel Density Estimation (KDE) to build a probabilistic surrogate model.

### Strategy

Unlike a standard Bayesian optimization approach that uses a Gaussian Process (GP) to model the objective function, TPE inverts the modeling process. It focuses on modeling the conditional probability of hyperparameters $x$ given the observed objective score $y$, denoted as $P(x|y)$. This is achieved by defining two density functions based on a threshold $y^{*}$: $l(x)=P(x|y<y^{*})$, the probability density function (PDF) for the "good" hyperparameters that resulted in a score better than $y^{*}$. $g(x)=P(x|y\ge y^{*})$, the PDF for the "bad" hyperparameters that performed worse than $y^{*}$. The threshold $y^{*}$ is typically chosen as a quantile of the observed scores, controlled by a parameter $\gamma$. The algorithm then uses these two densities to construct an acquisition function that maximizes the ratio $l(x)/g(x)$, identifying areas where "good" hyperparameters are dense and "bad" ones are sparse.

### Key Mathematical Steps

1. Probability density estimation with Parzen windows:
TPE uses Kernel Density Estimation (KDE), also known as Parzen windows, to construct the density functions $l(x)$ and $g(x)$. The KDE for a set of data points $\{x_{1},\dots ,x_{n}\}$ is defined as:
$\^{f}_{h}(x)=\frac{1}{n}\sum _{i=1}^{n}K_{h}(x-x_{i})$
Here, $K_{h}(u)=\frac{1}{h}K(\frac{u}{h})$ is a kernel function (often a Gaussian) with bandwidth $h$. TPE fits a KDE to the "good" points to create $l(x)$ and another to the "bad" points to create $g(x)$.

2. Acquisition function and Expected Improvement (EI):
The next set of hyperparameters to test, $x_{next}$, is selected by maximizing the acquisition function. The acquisition function in TPE is based on the Expected Improvement (EI), which is approximated by a sampling procedure.
$EI_{y^{*}}(x)=\int _{-\infty }^{y^{*}}(y^{*}-y)P(y|x)dy$
Since TPE models $P(x|y)$ instead of $P(y|x)$, this expectation is calculated by sampling from $l(x)$ and $g(x)$:
$EI(x)\propto \frac{l(x)}{g(x)}$
The maximization of this ratio is computationally simple when using TPE's tree-structured representation of the search space.

3. Conditional search space:
The "tree-structured" aspect of TPE allows it to handle complex, nested search spaces. For instance, a neural network might have a hyperparameter for the optimizer type, and if that optimizer is Adam, specific parameters like the learning rate become relevant. TPE represents these dependencies in a tree-like structure, allowing it to define distributions for hyperparameters conditionally.

### Parameters

* **$\gamma$**: The quantile threshold that divides the observed trials into the "good" and "bad" groups. For example, a value of 0.25 means the top 25% of performing trials are considered "good".

* **n_startup_trials**: The number of initial, random trials to perform before the TPE algorithm begins modeling. A larger value can give a better initial approximation of the search space.

* **n_ei_candidates**: The number of candidates sampled from the $l(x)$ distribution to find the one with the highest expected improvement.

* **prior_weight**: A parameter that controls how much the algorithm should respect the initial hyperparameter priors relative to the observed data. A higher weight gives more emphasis to the initial search space definition.

* **multivariate**: An optional flag in implementations like Optuna that allows the KDE to model dependencies between parameters, rather than assuming independence.


### Workflow

* Initialization: Run n_startup_trials using random hyperparameters to get initial objective function evaluations.

* Partitioning: Based on the results, divide the evaluated hyperparameters into a "good" set (e.g., top $\gamma$ quantile) and a "bad" set.

* Density modeling: Fit a Parzen estimator (KDE) to both the "good" set, defining $l(x)$, and the "bad" set, defining $g(x)$.

* Proposal: Sample a large number of candidate hyperparameters from the "good" distribution $l(x)$. Calculate the ratio $l(x)/g(x)$ for each candidate and select the one that maximizes this ratio.

* Evaluation: Evaluate the objective function with the new, proposed hyperparameters.

* Iteration: Add the new result to the historical data and repeat from step 2 until the budget is exhausted.

* Return: The set of hyperparameters that yielded the best objective score is returned

### Pseudocode

```text
// Initialization
Set a random seed
Generate n_startup_trials of random hyperparameters and evaluate them
history = {(x_i, f(x_i))} for i=1 to n_startup_trials

// Iterative Optimization Loop
for n = n_startup_trials + 1 to N_budget:
    // Partition Trials
    y_threshold = quantile(scores_in_history, gamma)
    good_trials = {(x_i, f(x_i))} where f(x_i) < y_threshold
    bad_trials = {(x_i, f(x_i))} where f(x_i) >= y_threshold

    // Model Distributions
    Fit Parzen Estimator l(x) to hyperparameters from good_trials
    Fit Parzen Estimator g(x) to hyperparameters from bad_trials

    // Propose New Hyperparameters
    candidates = sample(l(x), size=n_ei_candidates)
    next_x = argmax_{x in candidates} [l(x) / g(x)]

    // Evaluate Objective
    next_y = f(next_x)

    // Update History
    history.add((next_x, next_y))

// Return Best Result
Return x_best in history that minimizes f(x)
```

---

## Random Search

Random Search replaces the exhaustive enumeration of all combinations by selecting them randomly. This can be simply applied to the discrete setting described above, but also generalizes to continuous and mixed spaces. A benefit over grid search is that random search can explore many more values than grid search could for continuous hyperparameters. It can outperform Grid search, especially when only a small number of hyperparameters affects the final performance of the machine learning algorithm. In this case, the optimization problem is said to have a low intrinsic dimensionality. Random Search is also embarrassingly parallel, and additionally allows the inclusion of prior knowledge by specifying the distribution from which to sample. Despite its simplicity, random search remains one of the important base-lines against which to compare the performance of new hyperparameter optimization methods.

### Pseudocode

```text
best_score = -∞
best_params = None
For iteration = 1 to N:
    Randomly sample hyperparameters from their ranges
    Train model with these hyperparameters
    Compute validation score
    If score > best_score:
        best_score = score
        best_params = sampled hyperparameters
Return best_params (and best_score)
```

---

## Grid Search

The traditional method for hyperparameter optimization has been grid search, or a parameter sweep, which is simply an exhaustive searching through a manually specified subset of the hyperparameter space of a learning algorithm. A grid search algorithm must be guided by some performance metric, typically measured by cross-validation on the training set[6] or evaluation on a hold-out validation set.

Since the parameter space of a machine learner may include real-valued or unbounded value spaces for certain parameters, manually set bounds and discretization may be necessary before applying grid search.

For example, a typical soft-margin SVM classifier equipped with an RBF kernel has at least two hyperparameters that need to be tuned for good performance on unseen data: a regularization constant C and a kernel hyperparameter γ. Both parameters are continuous, so to perform grid search, one selects a finite set of "reasonable" values for each, say :

C ∈ { 10 , 100 , 1000 }
γ ∈ { 0.1 , 0.2 , 0.5 , 1.0 }

Grid search then trains an SVM with each pair (C, γ) in the Cartesian product of these two sets and evaluates their performance on a held-out validation set (or by internal cross-validation on the training set, in which case multiple SVMs are trained per pair). Finally, the grid search algorithm outputs the settings that achieved the highest score in the validation procedure.

Grid search suffers from the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality), but is often embarrassingly parallel because the hyperparameter settings it evaluates are typically independent of each other.

### Pseudocode

```text
best_score = -∞
best_params = None
For each combination of hyperparameters in defined grid:
    Train model with these hyperparameters
    Compute validation score (e.g. via cross-validation)
    If score > best_score:
        best_score = score
        best_params = current combination
Return best_params (and best_score)
```
