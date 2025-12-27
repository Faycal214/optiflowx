from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from optiflowx.core import SearchSpace, ModelWrapper
import random


class RandomForestConfig:
    """Configuration class for Random Forest model optimization.

    Defines the hyperparameter search space and preprocessing logic
    for RandomForestClassifier before model initialization.
    """

    name = "random_forest"

    @staticmethod
    def build_search_space():
        """Construct the hyperparameter search space for RandomForestClassifier.

        Returns:
            SearchSpace: Search space containing tunable parameters such as
                number of estimators, depth, and sampling behavior.
        """
        s = SearchSpace()
        s.add("n_estimators", "discrete", [20, 100])
        s.add("max_depth", "discrete", [2, 20])
        s.add("min_samples_split", "discrete", [2, 10])
        s.add("min_samples_leaf", "discrete", [1, 5])
        s.add("bootstrap", "categorical", [True, False])
        s.add("max_features", "categorical", ["sqrt", "log2", None])
        return s

    @staticmethod
    def get_wrapper(task_type: str = "classification"):
        """Return a model wrapper for RandomForest.

        Args:
            task_type (str): "classification" or "regression". Determines which
                sklearn estimator is wrapped.

        Returns:
            ModelWrapper: Wrapper object integrating RandomForest estimator.
        """
        model_cls = (
            RandomForestClassifier if task_type == "classification" else RandomForestRegressor
        )
        return ModelWrapper(model_cls, preprocess=RandomForestConfig._preprocess_params)

    @staticmethod
    def _preprocess_params(params):
        """Preprocess parameters before model instantiation.

        Ensures valid parameter combinations for RandomForestClassifier,
        particularly regarding bootstrap sampling.

        Args:
            params (dict): Dictionary of hyperparameters.

        Returns:
            dict: Processed parameters with adjusted 'max_samples' field.
        """
        if not params["bootstrap"]:
            params["max_samples"] = None
        else:
            params["max_samples"] = random.uniform(0.3, 1.0)
        return params
