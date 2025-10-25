from sklearn.neural_network import MLPClassifier
from optiflowx.core.search_space import SearchSpace
from optiflowx.core.model_wrapper import ModelWrapper


class MLPConfig:
    """Configuration for Multi-Layer Perceptron classifier."""

    name = "mlp"

    @staticmethod
    def build_search_space():
        """Define the hyperparameter search space for MLPClassifier.

        Returns:
            SearchSpace: Search space containing architecture and training parameters.
        """
        s = SearchSpace()
        s.add(
            "hidden_layer_sizes",
            "categorical",
            [(50,), (100,), (200,), (100, 50), (200, 100, 50), (500, 250, 100)],
        )
        s.add("activation", "categorical", ["relu", "tanh", "logistic", "identity"])
        s.add("solver", "categorical", ["adam", "sgd", "lbfgs"])
        s.add("alpha", "continuous", [1e-6, 1e-1], log=True)
        s.add("learning_rate", "categorical", ["constant", "invscaling", "adaptive"])
        s.add("learning_rate_init", "continuous", [1e-5, 1e-1], log=True)
        s.add("batch_size", "discrete", [8, 512])
        s.add("momentum", "continuous", [0.1, 0.99])
        s.add("n_iter_no_change", "discrete", [5, 100])
        s.add("beta_1", "continuous", [0.7, 0.99])
        s.add("beta_2", "continuous", [0.8, 0.999])
        s.add("early_stopping", "categorical", [True, False])
        return s

    @staticmethod
    def get_wrapper():
        """Return model wrapper for MLPClassifier.

        Returns:
            ModelWrapper: Wrapper for the neural network classifier.
        """
        return ModelWrapper(MLPClassifier)
