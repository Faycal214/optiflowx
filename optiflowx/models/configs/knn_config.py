from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from optiflowx.core import SearchSpace, ModelWrapper


class KNNConfig:
    """Configuration for K-Nearest Neighbors classifier."""

    name = "knn"

    @staticmethod
    def build_search_space():
        """Define the hyperparameter search space for KNeighborsClassifier.

        Returns:
            SearchSpace: Search space containing neighbors, distance metrics, and algorithms.
        """
        s = SearchSpace()
        s.add("n_neighbors", "discrete", list(range(1, 31)))
        s.add("weights", "categorical", ["uniform", "distance"])
        s.add("algorithm", "categorical", ["auto", "ball_tree", "kd_tree", "brute"])
        s.add("leaf_size", "discrete", [5, 150])
        s.add("p", "discrete", [1, 5])
        s.add(
            "metric",
            "categorical",
            ["minkowski", "manhattan", "euclidean", "chebyshev"],
        )
        return s

    @staticmethod
    def get_wrapper(task_type: str = "classification"):
        """Return model wrapper for KNeighbors (classifier or regressor).

        Args:
            task_type (str): "classification" or "regression".

        Returns:
            ModelWrapper: Wrapper for the KNN model.
        """
        model_cls = KNeighborsClassifier if task_type == "classification" else KNeighborsRegressor
        return ModelWrapper(model_cls)
