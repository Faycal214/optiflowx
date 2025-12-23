# Algorithms in OptiFlowX
---
sidebar_label: Algorithms (index)
---

# Algorithms

This page has been refactored into focused subpages under `docs/algorithms/` for each optimizer (GA, PSO, SA, Bayesian, TPE, Random Search, Grid Search, GWO, ACO). See the Algorithms index or the sidebar to navigate.

You can find the split pages in `docs/algorithms/` (one file per algorithm) — all original content is preserved and reorganized for readability.

<img src="https://upload.wikimedia.org/wikipedia/commons/e/ec/ParticleSwarmArrowsAnimation.gif" alt="drawing" width="500"/>

PSO is originally attributed to Kennedy, Eberhart and Shi and was first intended for simulating social behaviour, as a stylized representation of the movement of organisms in a bird flock or fish school. The algorithm was simplified and it was observed to be performing optimization. The book by Kennedy and Eberhart describes many philosophical aspects of PSO and swarm intelligence. An extensive survey of PSO applications is made by Poli. In 2017, a comprehensive review on theoretical and experimental works on PSO has been published by Bonyadi and Michalewicz.

PSO is a metaheuristic as it makes few or no assumptions about the problem being optimized and can search very large spaces of candidate solutions. Also, PSO does not use the [gradient](https://en.wikipedia.org/wiki/Gradient) of the problem being optimized, which means PSO does not require that the optimization problem be [differentiable](https://en.wikipedia.org/wiki/Differentiable_function) as is required by classic optimization methods such as gradient descent and quasi-newton methods. However, metaheuristics such as PSO do not guarantee an optimal solution is ever found.

### Algorithm

A basic variant of the PSO algorithm works by having a population (called a swarm) of candidate solutions (called particles). These particles are moved around in the search-space according to a few simple formulae. The movements of the particles are guided by their own best-known position in the search-space as well as the entire swarm's best-known position. When improved positions are being discovered these will then come to guide the movements of the swarm. The process is repeated and by doing so it is hoped, but not guaranteed, that a satisfactory solution will eventually be discovered.

Formally, let $f: \mathbb{R}^n \rightarrow \mathbb{R}$ be the cost function which must be minimized. The function takes a candidate solution as an argument in the form of a vector of real numbers and produces a real number as output which indicates the objective function value of the given candidate solution. The gradient of $f$ is not known. The goal is to find a solution $a$ for which $f(a) \leq f(b)$ for all $b$ in the search-space, which would mean $a$ is the global minimum.

Let $S$ be the number of particles in the swarm, each having a position $x_i \in \mathbb{R}^n$ in the search-space and a velocity $v_i \in \mathbb{R}^n$. Let $p_i$ be the best known position of particle $i$ and let $g$ be the best known position of the entire swarm. A basic PSO algorithm to minimize the cost function is then

```text
---
sidebar_label: Algorithms (index)
---

# Algorithms

This page was split into focused subpages under `docs/algorithms/` to improve navigation and readability. Please use the per-algorithm pages for full content:

- [Particle Swarm Optimization (PSO)](./algorithms/pso)
- [Simulated Annealing (SA)](./algorithms/simulated-annealing)
- [Bayesian Optimization](./algorithms/bayesian-optimization)
- [Tree-structured Parzen Estimator (TPE)](./algorithms/tpe)
- [Random Search](./algorithms/random-search)
- [Grid Search](./algorithms/grid-search)
- [Grey Wolf Optimization (GWO)](./algorithms/grey-wolf-optimization)
- [Ant Colony Optimization (ACO)](./algorithms/ant-colony-optimization)

The original consolidated content has been preserved within the subpages.

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

## Grey Wolf Optimization (GWO)

**Grey Wolf Optimization** (GWO) is a nature-inspired metaheuristic algorithm that mimics the leadership hierarchy and hunting behavior of [grey wolves](https://en.wikipedia.org/wiki/Grey_wolf) in the wild. It was introduced by Seyedali Mirjalili in 2014 as a [swarm intelligence](https://en.wikipedia.org/wiki/Swarm_intelligence)-based technique for solving optimization problems. The algorithm is designed based on the social dominance structure of grey wolves, where the pack is led by an alpha, followed by beta and delta wolves, while omegas hold the lowest rank. This leadership hierarchy plays a crucial role in guiding the search for optimal solutions by balancing exploration and exploitation. The alpha wolves guide the hunt, while the beta and delta wolves assist in refining the movement and decision-making process.

In GWO, optimization is performed through three main steps: encircling prey, hunting, and attacking or diverging towards new solutions. Encircling is the process where wolves adjust their positions relative to the best solutions found so far. Hunting involves the collective effort of alpha, beta, and delta wolves, which estimate the prey’s location and guide the pack toward optimal solutions. Finally, the attack phase focuses on intensifying the search by reducing the distance between wolves and the best-known solution, ensuring convergence. If the solution space needs further exploration, the wolves diverge, helping prevent premature convergence to local optima.

One of the significant advantages of GWO is its simplicity and ability to handle complex optimization problems with fewer control parameters than other metaheuristic algorithms like genetic algorithms or particle swarm optimization. Its efficiency in finding global optima makes it suitable for a wide range of applications, including power system optimization, feature selection in machine learning, and structural engineering. Additionally, its ability to balance exploration and exploitation helps maintain diversity in the search process, reducing the likelihood of getting stuck in local minima.

In power system applications, GWO has been widely used for optimizing network configurations, enhancing resilience, and reducing operational costs. For instance, in resilient distribution network design, GWO helps allocate feeder routing, substation facilities, and reinforcement strategies to mitigate the impact of physical attacks or natural disasters. By considering both economic and technical constraints, the algorithm finds an optimal trade-off between resilience and cost-effectiveness. Its adaptability to large-scale problems makes it particularly useful in complex power grid scenarios where multiple variables must be optimized simultaneously.

Despite its advantages, GWO has some limitations, such as its reliance on the initial population and the potential for slow convergence in high-dimensional spaces. To improve its performance, researchers have proposed hybrid approaches that integrate GWO with other optimization techniques, such as fuzzy logic, artificial neural networks, or differential evolution. These modifications aim to enhance solution accuracy, speed, and adaptability in dynamic environments. Overall, GWO remains a powerful and flexible optimization tool with broad applications in engineering, machine learning, and power system resilience.

### Article : [Grey Wolf Optimization PDF](https://www.researchgate.net/publication/318185247_Grey_Wolf_Optimization_GWO_Algorithm)

---

## Ant colony optimization (ACO)

In computer science and operations research, the **ant colony optimization algorithm (ACO)** is a probabilistic technique for solving computational problems that can be reduced to finding good paths through graphs. Artificial ants represent multi-agent methods inspired by the behavior of real ants. The pheromone-based communication of biological ants is often the predominant paradigm used. Combinations of artificial ants and local search algorithms have become a preferred method for numerous optimization tasks involving some sort of graph, e.g., vehicle routing and internet routing. As an example, ant colony optimization is a class of optimization algorithms modeled on the actions of an ant colony. Artificial 'ants' (e.g. simulation agents) locate optimal solutions by moving through a parameter space representing all possible solutions. Real ants lay down pheromones to direct each other to resources while exploring their environment. The simulated 'ants' similarly record their positions and the quality of their solutions, so that in later simulation iterations more ants locate better solutions. One variation on this approach is the bees algorithm, which is more analogous to the foraging patterns of the honey bee, another social insect. This algorithm is a member of the ant colony algorithms family, in swarm intelligence methods, and it constitutes some metaheuristic optimizations. Initially proposed by Marco Dorigo in 1992 in his PhD thesis, the first algorithm was aiming to search for an optimal path in a graph, based based on the behavior of ants seeking a path between their colony and a source of food. The original idea has since diversified to solve a wider class of numerical problems, and as a result, several problems have emerged, drawing on various aspects of the behavior of ants. From a broader perspective, ACO performs a model-based search and shares some similarities with estimation of distribution algorithms.

### Overview

In the natural world, ants of some species (initially) wander randomly, and upon finding food return to their colony while laying down pheromone trails. If other ants find such a path, they are likely to stop travelling at random and instead follow the trail, returning and reinforcing it if they eventually find food (see Ant communication). Over time, however, the pheromone trail starts to evaporate, thus reducing its attractive strength. The more time it takes for an ant to travel down the path and back again, the more time the pheromones have to evaporate. A short path, by comparison, is marched over more frequently, and thus the pheromone density becomes higher on shorter paths than longer ones. Pheromone evaporation also has the advantage of avoiding the convergence to a locally optimal solution. If there were no evaporation at all, the paths chosen by the first ants would tend to be excessively attractive to the following ones. In that case, the exploration of the solution space would be constrained. The influence of pheromone evaporation in real ant systems is unclear, but it is very important in artificial systems. The overall result is that when one ant finds a good (i.e., short) path from the colony to a food source, other ants are more likely to follow that path, and positive feedback eventually leads to many ants following a single path. The idea of the ant colony algorithm is to mimic this behavior with "simulated ants" walking around the graph representing the problem to be solved.

### Ambient networks of intelligent objects

New concepts are required since "intelligence" is no longer centralized but can be found throughout all minuscule objects. Anthropocentric concepts have been known to lead to the production of IT systems in which data processing, control units and calculating power are centralized. These centralized units have continually increased their performance and can be compared to the human brain. The model of the brain has become the ultimate vision of computers. Ambient networks of intelligent objects and, sooner or later, a new generation of information systems that are even more diffused and based on nanotechnology, will profoundly change this concept. Small devices that can be compared to insects do not possess a high intelligence on their own. Indeed, their intelligence can be classed as fairly limited. It is, for example, impossible to integrate a high performance calculator with the power to solve any kind of mathematical problem into a biochip that is implanted into the human body or integrated in an intelligent tag designed to trace commercial articles. However, once those objects are interconnected they develop a form of intelligence that can be compared to a colony of ants or bees. In the case of certain problems, this type of intelligence can be superior to the reasoning of a centralized system similar to the brain.

Nature offers several examples of how minuscule organisms, if they all follow the same basic rule, can create a form of collective intelligence on the macroscopic level. Colonies of social insects perfectly illustrate this model which greatly differs from human societies. This model is based on the cooperation of independent units with simple and unpredictable behavior. They move through their surrounding area to carry out certain tasks and only possess a very limited amount of information to do so. A colony of ants, for example, represents numerous qualities that can also be applied to a network of ambient objects. Colonies of ants have a very high capacity to adapt themselves to changes in the environment, as well as great strength in dealing with situations where one individual fails to carry out a given task. This kind of flexibility would also be very useful for mobile networks of objects which are perpetually developing. Parcels of information that move from a computer to a digital object behave in the same way as ants would do. They move through the network and pass from one node to the next with the objective of arriving at their final destination as quickly as possible.

### Artificial pheromone system

Pheromone-based communication is one of the most effective ways of communication which is widely observed in nature. Pheromone is used by social insects such as bees, ants and termites; both for inter-agent and agent-swarm communications. Due to its feasibility, artificial pheromones have been adopted in multi-robot and swarm robotic systems. Pheromone-based communication was implemented by different means such as chemical or physical (RFID tags, light, sound) ways. However, those implementations were not able to replicate all the aspects of pheromones as seen in nature. Using projected light was presented in a 2007 IEEE paper by Garnier, Simon, et al. as an experimental setup to study pheromone-based communication with micro autonomous robots. Another study presented a system in which pheromones were implemented via a horizontal LCD screen on which the robots moved, with the robots having downward facing light sensors to register the patterns beneath them.

### Algorithm and formula

In the ant colony optimization algorithms, an artificial ant is a simple computational agent that searches for good solutions to a given optimization problem. To apply an ant colony algorithm, the optimization problem needs to be converted into the problem of finding the shortest path on a weighted graph. In the first step of each iteration, each ant stochastically constructs a solution, i.e. the order in which the edges in the graph should be followed. In the second step, the paths found by the different ants are compared. The last step consists of updating the pheromone levels on each edge.

```text
procedure ACO_MetaHeuristic is
    while not terminated do
        generateSolutions()
        daemonActions()
        pheromoneUpdate()
    repeat
end procedure
```

### Edge selection

Each ant needs to construct a solution to move through the graph. To select the next edge in its tour, an ant will consider the length of each edge available from its current position, as well as the corresponding pheromone level. At each step of the algorithm, each ant moves from a state x {\displaystyle x} to state y {\displaystyle y}, corresponding to a more complete intermediate solution. Thus, each ant k {\displaystyle k} computes a set A k ( x ) {\displaystyle A_{k}(x)} of feasible expansions to its current state in each iteration, and moves to one of these in probability.

For ant k {\displaystyle k}, the probability p x y k {\displaystyle p_{xy}^{k}} of moving from state x {\displaystyle x} to state y {\displaystyle y} depends on the combination of two values, the attractiveness η x y {\displaystyle \eta_{xy}} of the move, as computed by some heuristic indicating the a priori desirability of that move and the trail level τ x y {\displaystyle \tau _{xy}} of the move, indicating how proficient it has been in the past to make that particular move. The trail level represents a posteriori indication of the desirability of that move.

In general, the k {\displaystyle k}th ant moves from state x {\displaystyle x} to state y {\displaystyle y} with probability

p_{xy}^{k} = ((τ_xy^α)(η_xy^β)) / ( Σ_{z ∈ allowed_y} (τ_xz^α)(η_xz^β) )

where τ x y is the amount of pheromone deposited for transition from state x to y, α ≥ 0 controls the influence of τ x y, η x y is the desirability of the move (typically 1 / d_xy), and β ≥ 1 controls the influence of η x y.

### Pheromone update

Trails are usually updated when all ants have completed their solution, increasing or decreasing trail levels depending on "good" or "bad" solutions. A global rule:

τ_xy ← (1 − ρ) τ_xy + Σ_k Δτ_xy^k

where ρ is the evaporation coefficient, and Δτ_xy^k is typically:

Δτ_xy^k = { Q / L_k if ant k uses edge xy
0 otherwise }

### Common extensions

#### Ant system (AS)

The first ACO algorithm, corresponding to the one above.

#### Ant colony system (ACS)

Modifies AS in three aspects:

- Biased edge selection toward exploitation.
- Local pheromone updates during solution construction.
- Only the best ant updates trails globally.

#### Elitist ant system

Global best solution deposits pheromone each iteration.

#### Max-min ant system (MMAS)

Controls max/min pheromone per trail, restricts deposition to the best ant, reinitializes trails at stagnation.

#### Rank-based ant system (ASrank)

Solutions ranked by length; best few update pheromone with weighted contributions.

#### Parallel ant colony optimization (PACO)

Partitions ants into groups with communication strategies.

#### Continuous orthogonal ant colony (COAC)

Uses orthogonal design and adaptive radius for strong global search.

#### Recursive ant colony optimization

Divides domain recursively into subdomains, promoting best solutions.

### Convergence

For some versions of the algorithm, it is possible to prove that it is convergent (i.e., it is able to find the global optimum in finite time). The first evidence of convergence for an ant colony algorithm was made in 2000, the graph-based ant system algorithm, and later on for the ACS and MMAS algorithms. Like most metaheuristics, it is very difficult to estimate the theoretical speed of convergence. A performance analysis of a continuous ant colony algorithm with respect to its various parameters (edge selection strategy, distance measure metric, and pheromone evaporation rate) showed that its performance and rate of convergence are sensitive to the chosen parameter values, and especially to the value of the pheromone evaporation rate. In 2004, Zlochin and his colleagues showed that ACO-type algorithms are closely related to stochastic gradient descent, Cross-entropy method and estimation of distribution algorithm. They proposed an umbrella term "Model-based search" to describe this class of metaheuristics.

### Article : [Ant Colony Optimization: Artificial Ants as a Computational Intelligence Technique PDF](https://web.archive.org/web/20120222061542/http://iridia.ulb.ac.be/IridiaTrSeries/IridiaTr2006-023r001.pdf)
