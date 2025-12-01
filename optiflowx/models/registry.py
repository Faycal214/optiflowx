# optiflowx/models/registry.py
"""
Central model registry.
It imports the config classes you already have in models/configs and exposes:
 - MODEL_REGISTRY dict
 - get_model_config(name) helper
"""

from optiflowx.models.configs.svc_config import SVCConfig
from optiflowx.models.configs.random_forest_config import RandomForestConfig
from optiflowx.models.configs.xgboost_config import XGBoostConfig
from optiflowx.models.configs.mlp_config import MLPConfig
from optiflowx.models.configs.decision_tree_config import DecisionTreeConfig
from optiflowx.models.configs.knn_config import KNNConfig
from optiflowx.models.configs.logistic_regression_config import LogisticRegressionConfig
from optiflowx.models.configs.custom_model_config import CustomModelConfig

# add imports only for config files that actually exist

MODEL_REGISTRY = {
    "svc": SVCConfig,
    "random_forest": RandomForestConfig,
    "xgboost": XGBoostConfig,
    "mlp": MLPConfig,
    "decision_tree": DecisionTreeConfig,
    "knn": KNNConfig,
    "logistic_regression": LogisticRegressionConfig,
    "custom_model": CustomModelConfig,
}


def get_model_config(name: str):
    """Return the config class for a model key (e.g., 'svc')."""
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
