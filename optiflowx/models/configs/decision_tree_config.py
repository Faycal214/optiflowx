from sklearn.tree import DecisionTreeClassifier
from optiflowx.core.search_space import SearchSpace
from optiflowx.core.model_wrapper import ModelWrapper


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
    def get_wrapper():
        """Return a model wrapper for DecisionTreeClassifier.

        Returns:
            ModelWrapper: Wrapper providing standardized model interface.
        """
        return ModelWrapper(DecisionTreeClassifier)
