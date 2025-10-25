from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from typing import Callable, Optional


class ModelWrapper:
    """Wrapper around a scikit-learn estimator for cross-validation and final fitting."""

    def __init__(self, model_class, preprocess: Optional[Callable] = None):
        """Initialize the model wrapper.

        Args:
            model_class: A scikit-learn estimator class (not instance).
            preprocess (Callable, optional): Function(params) -> params used to
                transform hyperparameters before model instantiation.
        """
        self.model_class = model_class
        self.preprocess = preprocess

    def train_and_score(
        self, params: dict, X, y, cv: int = 3, scoring: str = "accuracy"
    ) -> float:
        """Evaluate hyperparameters via cross-validation.

        Args:
            params (dict): Estimator hyperparameters.
            X: Feature matrix.
            y: Target vector.
            cv (int): Number of cross-validation folds.
            scoring (str): Scoring metric key for sklearn.

        Returns:
            float: Mean cross-validation score.
        """
        if self.preprocess:
            params = self.preprocess(params)

        model = self.model_class(**params)
        scores = cross_val_score(clone(model), X, y, cv=cv, scoring=scoring)
        return float(scores.mean())

    def fit_final(self, params: dict, X, y):
        """Train the final estimator on the full dataset.

        Args:
            params (dict): Final optimized hyperparameters.
            X: Feature matrix.
            y: Target vector.

        Returns:
            Fitted estimator: The trained scikit-learn model instance.
        """
        if self.preprocess:
            params = self.preprocess(params)
        model = self.model_class(**params)
        model.fit(X, y)
        return model
