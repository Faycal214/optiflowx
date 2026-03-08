# Table of Contents

1. [Simulated Annealing](#simulated-annealing)
2. [Bayesian Optimization](#bayesian-optimization)
3. [TPE](#tpe)
4. [Random Search](#random-search)
5. [Grid Search](#grid-search)
6. [GWO](#gwo)
7. [ACO](#aco)


## Simulated Annealing

Simulated Annealing (SA) is a probabilistic technique for approximating the global optimum of a given function. The algorithm uses a temperature parameter that decreases over time, allowing for a controlled exploration of the solution space.

### Pseudocode:
```
function SimulatedAnnealing(problem)
    currentSolution ← initialSolution(problem)
    currentEnergy ← Energy(currentSolution)
    T ← initialTemperature

    while not stoppingCondition() do
        newSolution ← neighbor(currentSolution)
        newEnergy ← Energy(newSolution)
        if newEnergy < currentEnergy then
            currentSolution ← newSolution
            currentEnergy ← newEnergy
        else
            probability ← exp((currentEnergy - newEnergy) / T)
            if random(0, 1) < probability then
                currentSolution ← newSolution
                currentEnergy ← newEnergy
        T ← updateTemperature(T)
    return currentSolution
```

### Mathematical Formula:

\[ E = E_0 e^{-kt} \]

---

## Bayesian Optimization

Bayesian Optimization is a strategy for the optimization of objective functions that are expensive to evaluate.

### Pseudocode:
```
function BayesianOptimization(objectiveFunc, bounds)
    gpModel ← initializeGaussianProcess()
    bestSolution ← NULL

    for iteration in range(maxIterations) do
        xNext ← acquireNextLocation(gpModel, bounds)
        yNext ← objectiveFunc(xNext)
        updateGaussianProcess(gpModel, xNext, yNext)
        bestSolution ← getBestSolution(gpModel)
    return bestSolution
```

### Mathematical Formula:

\[ y = f(x) + 	ext{noise} \],\quad f(x) 	ext{ is approximated by a Gaussian process}.

---

## TPE (Tree-structured Parzen Estimator)

TPE is an optimization algorithm based on a Bayesian optimization approach.

### Pseudocode:
```
function TPE(objectiveFunc, bounds)
    for iteration in range(maxIterations) do
        sample ← randomSample(bounds)
        evaluation ← objectiveFunc(sample)
        updateModel(sample, evaluation)
    return bestConfiguration()
```

### Mathematical Formula:

\[ p(x | y) = \frac{p(y | x) p(x)}{p(y)} \]

---

## Random Search

Random Search is a simple optimization algorithm that evaluates random combinations of hyperparameters.

### Pseudocode:
```
function RandomSearch(objectiveFunc, bounds)
    bestSolution ← NULL
    for iteration in range(maxEvaluations) do
        x ← randomSample(bounds)
        evaluation ← objectiveFunc(x)
        if bestSolution is NULL or evaluation < bestObjective then
            bestSolution ← x
            bestObjective ← evaluation
    return bestSolution
```

---

## Grid Search

Grid Search performs an exhaustive search over specified parameter values.

### Pseudocode:
```
function GridSearch(objectiveFunc, paramGrid)
    bestSolution ← NULL
    for params in paramGrid do
        evaluation ← objectiveFunc(params)
        if bestSolution is NULL or evaluation < bestObjective then
            bestSolution ← params
            bestObjective ← evaluation
    return bestSolution
```

---

## GWO (Gray Wolf Optimization)

GWO is a swarm intelligence optimization algorithm inspired by the social hierarchy and hunting mechanisms of gray wolves.

### Pseudocode:
```
function GWO(objectiveFunc, numWolves)
    initializeWolves(numWolves)
    while not stoppingCondition() do
        for wolf in wolves do
            evaluate(wolf)
        updatePositions()
    return bestSolution()
```

### Mathematical Formula:

\[ D = |C \cdot X_{rand} - X| \]

---

## ACO (Ant Colony Optimization)

ACO is a probabilistic technique used in computing to find optimal paths through graphs.

### Pseudocode:
```
function ACO(graph, ants)
    initializePheromones(graph)
    for iteration in range(maxIterations) do
        for ant in ants do
            path ← constructSolution(ant, graph)
            updatePheromones(path)
    return bestPath()
```

### Mathematical Formula:

\[ P_{ij} = \frac{\tau_{ij}^\alpha \eta_{ij}^\beta}{\sum_{k \in allowed} \tau_{ik}^\alpha \eta_{ik}^\beta} \]
