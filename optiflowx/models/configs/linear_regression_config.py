from sklearn.linear_model import LinearRegression
from optiflowx.core import SearchSpace, ModelWrapper


class LinearRegressionConfig:
    name = "linear_regression"

    @staticmethod
    def build_search_space():
        s = SearchSpace()
        s.add("fit_intercept", "categorical", [True, False])
        s.add("positive", "categorical", [True, False])
        s.add("copy_X", "categorical", [True, False])
        return s

    @staticmethod
    def get_wrapper():
        return ModelWrapper(LinearRegression)
