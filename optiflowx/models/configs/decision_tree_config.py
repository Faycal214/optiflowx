from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from optiflowx.core import SearchSpace, ModelWrapper


class DecisionTreeConfig:
    """Configuration class for Decision Tree model optimization.

    Defines the model name, its hyperparameter search space, and the wrapper
    used to integrate with the optimization engine.
    """

    name = "decision_tree"

    @staticmethod
    def build_search_space():
        """Construct the hyperparameter search space for DecisionTreeClassifier.

        Returns:
            SearchSpace: Configured search space with all tunable parameters.
        """
        s = SearchSpace()
        s.add("criterion", "categorical", ["gini", "entropy", "log_loss"])
        s.add("splitter", "categorical", ["best", "random"])
        s.add("max_depth", "discrete", [1, 300])
        s.add("min_samples_split", "continuous", [1e-4, 0.9], log=True)
        s.add("min_samples_leaf", "continuous", [1e-4, 0.5], log=True)
        s.add("max_features", "categorical", ["sqrt", "log2", None])
        s.add("ccp_alpha", "continuous", [0.0, 0.2], log=False)
        s.add("min_weight_fraction_leaf", "continuous", [0.0, 0.5])
        return s

    @staticmethod
    def get_wrapper(task_type: str = "classification"):
        """Return a model wrapper for Decision Tree.

        Args:
            task_type (str): "classification" or "regression".

        Returns:
            ModelWrapper: Wrapper for the appropriate Decision Tree estimator.
        """
        model_cls = DecisionTreeClassifier if task_type == "classification" else DecisionTreeRegressor
        return ModelWrapper(model_cls)
