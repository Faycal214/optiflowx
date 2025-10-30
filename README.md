# OptiFlowX

[![PyPI version](https://img.shields.io/pypi/v/optiflowx.svg)]([https://pypi.org/project/optiflowx/])
[![Python versions](https://img.shields.io/pypi/pyversions/optiflowx.svg)](https://pypi.org/project/optiflowx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/Faycal214/optiflowx/actions/workflows/test.yml/badge.svg)](https://github.com/Faycal214/optiflowx/actions)


**OptiFlowX** is an open-source framework for **hyperparameter optimization** using **combinatorial and metaheuristic algorithms**.
It enables systematic exploration of search spaces for machine learning models using advanced stochastic and hybrid optimization techniques.

---

## 🔍 Overview

Machine learning models rely on carefully tuned hyperparameters.
Conventional methods such as grid or random search are inefficient when dealing with discrete, mixed, or high-dimensional spaces.

**OptiFlowX** provides a modular ecosystem that leverages **combinatorial optimization** and **metaheuristics** (evolutionary, swarm-based, and probabilistic methods) to efficiently discover optimal configurations.

---

## ⚙️ Key Features

- Unified interface for multiple optimization algorithms.
- Works with discrete, categorical, and continuous parameter spaces.
- Includes multiple strategies:
  - Genetic Algorithm (GA)
  - Particle Swarm Optimization (PSO)
  - Bayesian Optimization
  - Tree-structured Parzen Estimator (TPE)
  - Random Search
  - Simulated Annealing
- Scalable, parallel execution support.
- Compatible with any ML framework (scikit-learn, PyTorch, TensorFlow, etc.).

---

## ⚗️ Installation

```bash
pip install optiflowx
```

or from source:

```bash
git clone https://github.com/Faycal214/optiflowx.git
cd optiflowx
pip install -e .
```

---

## 🧠 Example Usage

```python
from sklearn.datasets import make_classification
from optiflowx.optimizers.genetic import GeneticOptimizer
from optiflowx.optimizers.pso import PSOOptimizer
from optiflowx.optimizers.bayesian import BayesianOptimizer
from optiflowx.optimizers.tpe import TPEOptimizer
from optiflowx.optimizers.random_search import RandomSearchOptimizer
from optiflowx.optimizers.simulated_annealing import SimulatedAnnealingOptimizer
from optiflowx.models.configs.random_forest_config import RandomForestConfig


# 1. Generate sample dataset
X, y = make_classification(
    n_samples=100,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    random_state=42,
)

# 2. Load model configuration and search space
cfg = RandomForestConfig()
search_space = cfg.build_search_space()
model_class = cfg.get_wrapper().model_class

# 3. Define optimizers to test
optimizers = [
    ("pso", PSOOptimizer, {"n_particles": 10, "w": 0.7, "c1": 1.4, "c2": 1.4}),
    ("genetic", GeneticOptimizer, {"population": 10, "mutation_prob": 0.3}),
    ("bayesian", BayesianOptimizer, {"n_initial_points": 5}),
    ("tpe", TPEOptimizer, {"population_size": 10}),
    ("random_search", RandomSearchOptimizer, {"n_samples": 20}),
    (
        "simulated_annealing",
        SimulatedAnnealingOptimizer,
        {
            "population_size": 10,
            "initial_temp": 1.0,
            "cooling_rate": 0.9,
            "mutation_rate": 0.3,
        },
    ),
]

# 4. Run each optimizer and print results
for opt_name, opt_class, opt_params in optimizers:
    print(f"\n{'='*30}\nTesting optimizer: {opt_name}\n{'='*30}")
    optimizer = opt_class(
        search_space=search_space,
        metric="accuracy",
        model_class=model_class,
        X=X,
        y=y,
        **opt_params,
    )

    best_params, best_score = optimizer.run(max_iters=5)
    print(f"Result for {opt_name} → score={best_score:.4f}, params={best_params}")
```

---

## 🧪 Testing


```bash
pytest -v --maxfail=1 --disable-warnings
```

---

## 📚 Citation

```bash
@software{optiflowx,
  author = {Faycal, Billel},
  title = {OptiFlowX: Combinatorial Hyperparameter Optimization Framework},
  year = {2025},
  url = {https://github.com/Faycal214/optiflowx}
}
```
