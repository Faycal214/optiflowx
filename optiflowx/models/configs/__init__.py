"""
All model configuration classes.
Each config defines:
- model class reference
- hyperparameter search space
- wrapper getter for training and inference
"""

from .svc_config import SVCConfig
from .random_forest_config import RandomForestConfig
from .xgboost_config import XGBoostConfig
from .mlp_config import MLPConfig
from .decision_tree_config import DecisionTreeConfig
from .knn_config import KNNConfig
from .linear_regression_config import LinearRegressionConfig
from .logistic_regression_config import LogisticRegressionConfig

__all__ = [
    "SVCConfig",
    "RandomForestConfig",
    "XGBoostConfig",
    "MLPConfig",
    "DecisionTreeConfig",
    "KNNConfig",
    "LinearRegressionConfig",
    "LogisticRegressionConfig",
]
