---
sidebar_label: Particle Swarm Optimization (PSO)
---

## Particle Swarm Optimization (PSO)

PSO is a population-based metaheuristic that moves a swarm of particles through the search-space using velocity updates influenced by local and global bests.

### Essentials
- Particles with positions and velocities
- Personal best (p_i) and global best (g)
- Velocity update blends inertia, cognitive and social terms

### Algorithm (summary)
```text
Initialize particles
For each iteration:
  Update velocities
  Update positions
  Update p_i and g
Return g
```

### Parameters
- Particles (20–200)
- Inertia weight (w)
- Cognitive/social coefficients (φ_p, φ_g)

### Pros / Cons
- Pros: simple, parallelizable
- Cons: premature convergence risk
