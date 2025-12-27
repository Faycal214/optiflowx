from sklearn.linear_model import LogisticRegression
from optiflowx.core import SearchSpace, ModelWrapper


class LogisticRegressionConfig:
    """Configuration for Logistic Regression classifier."""

    name = "logistic_regression"

    @staticmethod
    def build_search_space():
        """Define the hyperparameter search space for LogisticRegression.

        Returns:
            SearchSpace: Search space defining solver, penalty, and regularization settings.
        """
        s = SearchSpace()
        s.add("C", "continuous", [1e-5, 1e4], log=True)
        s.add("penalty", "categorical", ["l1", "l2", "elasticnet", "none"])
        s.add("solver", "categorical", ["liblinear", "lbfgs", "newton-cg", "saga"])
        s.add("max_iter", "discrete", [100, 10000])
        s.add("fit_intercept", "categorical", [True, False])
        s.add("class_weight", "categorical", [None, "balanced"])
        s.add("l1_ratio", "continuous", [0.0, 1.0])
        s.add("tol", "continuous", [1e-6, 1e-2], log=True)
        return s

    @staticmethod
    def get_wrapper():
        """Return model wrapper for LogisticRegression.

        Returns:
            ModelWrapper: Wrapper for the logistic regression model.
        """
        return ModelWrapper(LogisticRegression)
