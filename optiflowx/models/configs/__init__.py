"""
All model configuration classes.
Each config defines:
"""

from .svc_config import SVCConfig
from .random_forest_config import RandomForestConfig
from .mlp_config import MLPConfig
from .decision_tree_config import DecisionTreeConfig
from .knn_config import KNNConfig
from .linear_regression_config import LinearRegressionConfig
from .logistic_regression_config import LogisticRegressionConfig
from .custom_model_config import CustomModelConfig

# Optional imports
try:
    from .xgboost_config import XGBoostConfig
except Exception:
    XGBoostConfig = None  # XGBoost optional



"""
All model configuration classes.
Each config defines:
"""
__all__ = [
    "SVCConfig",
    "RandomForestConfig",
    "MLPConfig",
    "DecisionTreeConfig",
    "KNNConfig",
    "LinearRegressionConfig",
    "LogisticRegressionConfig",
    "CustomModelConfig",
]

# Add XGBoostConfig only if available
if XGBoostConfig is not None:
    __all__.append("XGBoostConfig")
