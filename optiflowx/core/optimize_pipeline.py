import time
import json
from tqdm import tqdm
from sklearn.model_selection import cross_val_score
import numpy as np
import logging


def optimize_model(model, param_grid, X, y, cv=5, n_jobs=-1, verbose=True):
    """Perform manual optimization over a set of parameter configurations.

    Evaluates multiple hyperparameter combinations using cross-validation,
    reports progress, logs results, and saves a JSON summary.

    Args:
        model: A scikit-learn compatible estimator.
        param_grid (list[dict]): List of parameter combinations to test.
        X: Feature matrix.
        y: Target vector.
        cv (int): Number of cross-validation folds.
        n_jobs (int): Number of parallel workers (-1 for all cores).
        verbose (bool): Whether to print iteration logs.

    Returns:
        tuple: (best_result, results)
            - best_result (dict): The best performing parameter configuration.
            - results (list[dict]): All configurations with scores and timings.
    """
    results = []
    start_time = time.time()
    total_configs = len(param_grid)
    logging.info("Starting optimization on %d configurations...", total_configs)

    with tqdm(total=total_configs, desc="Optimization Progress") as pbar:
        for i, params in enumerate(param_grid):
            iter_start = time.time()
            model.set_params(**params)

            scores = cross_val_score(model, X, y, cv=cv, n_jobs=n_jobs)
            mean_score = np.mean(scores)
            iter_time = time.time() - iter_start

            results.append(
                {
                    "iteration": i + 1,
                    "params": params,
                    "mean_score": mean_score,
                    "iteration_time_sec": iter_time,
                }
            )

            if verbose:
                logging.info("[%d/%d] Score: %.4f | Time: %.2fs", i+1, total_configs, mean_score, iter_time)

            pbar.update(1)

    total_time = time.time() - start_time
    logging.info("\nOptimization complete in %.2f seconds.", total_time)

    best_result = max(results, key=lambda x: x["mean_score"])
    with open("optimization_log.json", "w") as f:
        json.dump(
            {
                "total_time_sec": total_time,
                "results": results,
                "best_result": best_result,
            },
            f,
            indent=4,
        )

    logging.info("Best score: %.4f", best_result["mean_score"])
    return best_result, results
