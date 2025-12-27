# optiflowx/models/registry.py
"""
Central model registry.
It imports the config classes in `models/configs` and exposes:
 - MODEL_REGISTRY dict
 - get_model_config(name) helper
"""

from optiflowx.models.configs import (
    SVCConfig,
    RandomForestConfig,
    XGBoostConfig,
    MLPConfig,
    DecisionTreeConfig,
    KNNConfig,
    LogisticRegressionConfig,
    CustomModelConfig,
)

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
