---
sidebar_label: Genetic Algorithm (GA)
---

## Genetic Algorithm (GA)

In computer science and operations research, a **genetic algorithm (GA)** is a metaheuristic inspired by the process of natural selection that belongs to the larger class of evolutionary algorithms (EA). Genetic algorithms are commonly used to generate high-quality solutions to optimization and search problems via biologically inspired operators such as selection, crossover, and mutation.

### Key concepts
- Population: a set of candidate solutions.
- Genotype/chromosome: encoding of a solution.
- Fitness: objective evaluation function.

### Workflow (scannable)
1. Initialize population
2. Evaluate fitness
3. Select parents
4. Recombine (crossover) and mutate
5. Form new generation
6. Repeat until termination

### Parameters & tuning
- Population size (typical: 50–500)
- Crossover rate
- Mutation rate

### Pseudocode
```text
Initialize population P with N random candidate solutions
Evaluate fitness f(x) for each x in P
For generation = 1 to G:
    Select parent solutions from P
    Apply crossover and mutation
    Evaluate offspring
    Form new population
Output best solution(s)
```

### When to use
- Discrete/combinatorial search
- Problems where global exploration is required

### Pros / Cons
- Pros: robust, flexible
- Cons: many hyperparameters, compute-heavy for large populations
