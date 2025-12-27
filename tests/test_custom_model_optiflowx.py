
"""
OptiFlowX Example: Custom ML Model Integration
------------------------------------------------
This example demonstrates how to:
 1. Define a custom ML model (SimpleLinearModel)
 2. Create a config class for it
 3. Register it in OptiFlowX
 4. Optimize its hyperparameters using OptiFlowX optimizers
This is suitable for mkdocs and user reference.
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from optiflowx.core import SearchSpace, ModelWrapper
from optiflowx.optimizers import (
    GeneticOptimizer,
    PSOOptimizer,
    BayesianOptimizer,
    TPEOptimizer,
    RandomSearchOptimizer,
    SimulatedAnnealingOptimizer,
    GreyWolfOptimizer,
    AntColonyOptimizer,
)

# 1. Define a real custom ML model (simple linear regression with regularization)

# Simple custom classification model: MajorityClassClassifier
class MajorityClassClassifier:
    def __init__(self, dummy_param=0):
        self.dummy_param = dummy_param
        self.majority_class = None

    def fit(self, X, y):
        # Find the majority class in y
        values, counts = np.unique(y, return_counts=True)
        self.majority_class = values[np.argmax(counts)]
        return self

    def predict(self, X):
        # Predict the majority class for all samples
        return np.full(shape=(len(X),), fill_value=self.majority_class)

# 2. Create a config class for the custom model

# Config for MajorityClassClassifier
class MajorityClassConfig:
    name = "majority_class"

    @staticmethod
    def build_search_space():
        s = SearchSpace()
        s.add("dummy_param", "discrete", [0, 1])
        return s

    @staticmethod
    def get_wrapper():
        return ModelWrapper(MajorityClassClassifier)


# 3. Generate classification dataset
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)

# 4. Use the config and search space
cfg = MajorityClassConfig()
search_space = cfg.build_search_space()
model_class = cfg.get_wrapper().model_class

# 5. Define optimizers to test
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
    ("grey_wolf", GreyWolfOptimizer, {"population_size": 10, "max_iters": 5}),
    ("ant_colony", AntColonyOptimizer, {"colony_size": 10, "max_iters": 5, "evaporation_rate": 0.2}),
]

# 6. Run each optimizer and print results
for opt_name, opt_class, opt_params in optimizers:
    print(f"\n{'='*30}\nTesting optimizer: {opt_name}\n{'='*30}")
    optimizer = opt_class(
        search_space=search_space,
        metric=accuracy_score,
        model_class=model_class,
        X=X,
        y=y,
        **opt_params,
    )
    best_params, best_score = optimizer.run(max_iters=5)
    print(f"Result for {opt_name} → score={best_score:.4f}, params={best_params}")
