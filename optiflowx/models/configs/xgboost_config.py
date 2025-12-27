from optiflowx.core import SearchSpace, ModelWrapper


class XGBoostConfig:
    """Configuration for XGBoost classifier/regressor."""

    name = "xgboost"

    @staticmethod
    def build_search_space():
        """Define the hyperparameter search space for XGBClassifier/XGBRegressor.

        Returns:
            SearchSpace: Search space with structural and regularization parameters.
        """
        s = SearchSpace()
        s.add("n_estimators", "discrete", [50, 100, 200])
        s.add("max_depth", "discrete", [3, 5, 7])
        s.add("learning_rate", "continuous", [1e-3, 0.3], log=True)
        s.add("subsample", "continuous", [0.5, 1.0])
        s.add("colsample_bytree", "continuous", [0.5, 1.0])
        s.add("gamma", "continuous", [0.0, 5.0])
        s.add("min_child_weight", "discrete", [1, 10])
        s.add("reg_lambda", "continuous", [0.0, 10.0])
        s.add("reg_alpha", "continuous", [0.0, 10.0])
        return s

    @staticmethod
    def get_wrapper(task_type: str = "classification"):
        """Return model wrapper for XGBoost classifier or regressor.

        Args:
            task_type (str): "classification" or "regression".

        Returns:
            ModelWrapper: Wrapper for the XGBoost estimator.
        """
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as e:
            raise ImportError(
                "xgboost is not installed. Install with: pip install optiflowx[xgboost]"
            ) from e

        model_cls = XGBClassifier if task_type == "classification" else XGBRegressor
        return ModelWrapper(model_cls)
