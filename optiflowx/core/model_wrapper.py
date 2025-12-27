from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import clone
from typing import Callable, Optional
from .metrics import get_metric


class ModelWrapper:
    """Wrapper around a scikit-learn estimator for cross-validation and final fitting."""

    def __init__(self, model_class, preprocess: Optional[Callable] = None, custom_metric: Optional[Callable] = None, task_type: str = "classification"):
        """Initialize the model wrapper.

        Args:
            model_class: A scikit-learn estimator class (not instance).
            preprocess (Callable, optional): Function(params) -> params used to
                transform hyperparameters before model instantiation.
        """
        self.model_class = model_class
        self.preprocess = preprocess
        self.custom_metric = custom_metric
        self.task_type = task_type

    def train_and_score(
        self,
        params: dict,
        X,
        y,
        cv: int = 3,
        scoring: str = "accuracy",
        custom_metric: Optional[Callable] = None,
        task_type: str = "classification",
    ) -> tuple[float, object]:
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

        # If a custom metric is provided (callable y_true, y_pred), perform
        # manual KFold cross-validation so we can call it directly. This also
        # avoids relying on sklearn's scorer wrapper which expects estimator
        # style callables.
        # Prefer the explicit custom_metric argument, else fall back to wrapper default
        use_custom = custom_metric if custom_metric is not None else self.custom_metric
        if use_custom is not None:
            metric_fn = get_metric(use_custom)
            kf = KFold(n_splits=max(2, cv), shuffle=True, random_state=0)
            scores = []
            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                m = clone(model)
                m.fit(X_train, y_train)
                preds = m.predict(X_test)
                s = metric_fn(y_test, preds)
                try:
                    s = float(s)
                except Exception:
                    raise ValueError("Custom metric must return a numeric score")
                scores.append(s)
            mean_score = float(sum(scores) / len(scores)) if scores else float("-inf")
            # Fit final model on full data for returning
            final_model = model
            final_model.fit(X, y)
            return mean_score, final_model

        # Otherwise use sklearn's cross_val_score with scoring string
        scores = cross_val_score(clone(model), X, y, cv=cv, scoring=scoring)
        final_model = model
        final_model.fit(X, y)
        return float(scores.mean()), final_model

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
