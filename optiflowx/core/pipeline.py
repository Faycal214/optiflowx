from .optimization_engine import OptimizationEngine
import logging


class MLPipeline:
    """High-level entry point for running model optimization pipelines."""

    def __init__(self, model_key, dataset, optimizer="genetic", metric="accuracy"):
        """Initialize a machine learning optimization pipeline.

        Args:
            model_key (str): Registered model identifier in MODEL_REGISTRY.
            dataset (tuple): Tuple `(X, y)` containing features and labels.
            optimizer (str, optional): Optimization algorithm key.
                Defaults to `"genetic"`.
            metric (str, optional): Evaluation metric name.
                Defaults to `"accuracy"`.
        """
        self.model_key = model_key
        self.dataset = dataset
        self.optimizer = optimizer
        self.metric = metric

    def train(self, max_iters=5):
        """Run the optimization loop for the selected model.

        Args:
            max_iters (int, optional): Maximum number of optimization iterations.
                Defaults to 5.

        Returns:
            tuple:
                - best_model: Fitted model instance with optimal parameters.
                - best_params (dict): Best hyperparameters found.
                - best_score (float): Best evaluation score achieved.
        """
        logging.info("[Pipeline] Starting optimization for model: %s", self.model_key)

        engine = OptimizationEngine(
            model_key=self.model_key,
            optimizer_key=self.optimizer,
            dataset=self.dataset,
            metric=self.metric,
        )

        best_model, best_params, best_score = engine.run(max_iters=max_iters)

        logging.info("[Pipeline] Optimization complete.")
        logging.info("[Pipeline] Best score: %.4f", best_score)
        logging.info("[Pipeline] Best parameters: %s", best_params)

        return best_model, best_params, best_score
