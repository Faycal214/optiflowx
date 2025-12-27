from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from optiflowx.core import SearchSpace, ModelWrapper


def build_svc_pipeline(**params):
    """Build an SVC pipeline with standard scaling.

    Args:
        **params: Hyperparameters for the SVC model.

    Returns:
        Pipeline: A scikit-learn pipeline with StandardScaler and SVC.
    """
    return Pipeline([("scaler", StandardScaler()), ("svc", SVC(**params))])


def build_svr_pipeline(**params):
    """Build an SVR pipeline with standard scaling for regression."""
    return Pipeline([("scaler", StandardScaler()), ("svr", SVR(**params))])


class SVCConfig:
    """Configuration for SVM (Support Vector Classifier)."""

    name = "svc"

    @staticmethod
    def build_search_space():
        """Define the hyperparameter search space for SVC.

        Returns:
            SearchSpace: Defined search space for SVC hyperparameters.
        """
        s = SearchSpace()
        s.add("C", "continuous", [1e-3, 1e3], log=True)
        s.add("kernel", "categorical", ["linear", "rbf", "poly", "sigmoid"])
        s.add("gamma", "continuous", [1e-4, 1], log=True)
        s.add("degree", "discrete", [2, 5])
        return s

    @staticmethod
    def get_wrapper(task_type: str = "classification"):
        """Return model wrapper for SVC/SVR pipeline based on task type.

        Args:
            task_type (str): "classification" or "regression".

        Returns:
            ModelWrapper: Wrapper integrating preprocessing and model creation.
        """
        return ModelWrapper(build_svc_pipeline if task_type == "classification" else build_svr_pipeline)
