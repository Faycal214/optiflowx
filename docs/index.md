# 🧠 Welcome to **OptiFlowX**

**OptiFlowX** is an open-source optimization framework built for **machine learning**, **operations research**, and **applied AI**.
It provides a unified interface for running, comparing, and tracking optimization workflows — from hyperparameter tuning to algorithmic benchmarking.

---

## 🔍 What is OptiFlowX?

Modern AI pipelines rely heavily on tuning parameters, model selection, and efficiency trade-offs.
OptiFlowX was designed to make this process **faster**, **modular**, and **transparent**.
It combines multiple optimization techniques — from **Bayesian Optimization**, **Genetic Algorithms**, and **Swarm Intelligence**, to **Gradient-based methods** — under a single consistent API.

Whether you are optimizing a machine learning model, a mathematical function, or a large-scale system, OptiFlowX adapts to your use case.

---

## ⚡ Key Features

| Category | Description |
|-----------|--------------|
| **Multi-Algorithm Optimization** | Access classical, heuristic, and Bayesian optimizers via a unified interface. |
| **Cross-Domain Support** | Works with ML, operations research, and simulation-based optimization problems. |
| **Parallel Execution** | Utilize CPU cores efficiently for multi-objective or multi-run optimization. |
| **Smart Tracking** | Automatically logs metrics, configurations, and results for reproducibility. |
| **Modular Design** | Extend with custom algorithms or integrate your own scoring functions. |
| **Lightweight Integration** | Compatible with Scikit-Learn, PyTorch, TensorFlow, and custom models. |

---

## 🧩 System Overview

OptiFlowX architecture is organized into three layers:

1. **Configuration Layer**
   Define your problem, search space, and objective.

2. **Optimization Engine**
   Run your chosen algorithm (Bayesian, GA, PSO, etc.) efficiently across trials.

3. **Evaluation Layer**
   Validate, benchmark, and record the best performing configuration.

## 🧠 Supported Algorithms

OptiFlowX supports a broad range of optimizers:

- **Bayesian Optimization (Gaussian Process, TPE)**
- **Genetic Algorithms (GA)**
- **Particle Swarm Optimization (PSO)**
- **Simulated Annealing**
- **Grid and Random Search**
- **Gradient-Based Optimization**
- **Hybrid Metaheuristics**

Each algorithm is fully configurable and can run standalone or as part of a multi-optimizer pipeline.

---

## 📦 Installation

Install the latest stable release from PyPI:

```bash
pip install optiflowx
```

Or install directly from the source (recommended for contributors):

```bash
git clone https://github.com/faycal214/optiflowx.git
cd optiflowx
pip install -e .
```

## 🧭 Example Usage

```python
# importing libraries
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from optiflowx.optimizers.genetic import GeneticOptimizer
from optiflowx.models.configs.random_forest_config import RandomForestConfig

# loading the data
X, y = load_iris(return_X_y=True)

# Spliting the data into train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

# Importing model configurations for building the search space and explore the model
cfg = RandomForestConfig()
search_space = cfg.build_search_space()
model_class = cfg.get_wrapper().model_class

# Creating the optimizer object
optimizer = GeneticOptimizer(
    search_space=search_space,
    metric="accuracy",
    model_class=model_class,
    X=X_train,
    y=y_train,
    population=10,
    mutation_prob=0.3
)

# Proceeding with the choosing optimizer
best_params, best_score = optimizer.run(max_iters=5)
print(f"Best parameters : {best_params}")

# Test the model with the optimal parameters
model = RandomForestClassifier(**best_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

## 🧭 Philosophy

> “Optimization is the art of efficiency — finding balance between exploration and precision.”

OptiFlowX bridges research and engineering to make optimization accessible, interpretable, and powerful for everyone.

## 📈 Project Goals

* Simplify hyperparameter optimization for applied ML.

* Offer transparent and reproducible experiment tracking.

* Build an extensible framework adaptable to future algorithms.

* Provide educational and research-friendly tooling for optimization studies.
