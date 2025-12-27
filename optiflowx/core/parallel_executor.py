from multiprocessing import Pool
from typing import List
from .base_optimizer import Candidate
import os
import logging
import pickle

# Optional dill fallback for serializing non-pickleable callables
try:
    import dill
except Exception:
    dill = None


def _reconstruct_metric(obj):
    """Reconstruct a possibly-serialized metric object.

    The parallel executor may receive either a callable or a dict containing
    serialized bytes and a flag indicating whether `dill` was used. Rebuild
    the callable accordingly.
    """
    if obj is None:
        return None
    # If we received a tuple (by value) from parent process, reconstruct
    if isinstance(obj, dict) and obj.get("__serialized__"):
        data = obj.get("data")
        use_dill = obj.get("use_dill", False)
        if use_dill:
            if dill is None:
                raise RuntimeError("dill is required to deserialize the provided metric")
            return dill.loads(data)
        return pickle.loads(data)
    # Otherwise it's a regular callable
    return obj


def _eval_candidate_worker(args):
    """Worker function for parallel evaluation of a candidate.

    Args:
        args (tuple): `(candidate, wrapper, X, y, scoring, custom_metric, task_type)`.

    Returns:
        Candidate: The candidate with updated `score` and optional fitted `model`.
    """
    candidate, wrapper, X, y, scoring, custom_metric_obj, task_type = args
    try:
        custom_metric = _reconstruct_metric(custom_metric_obj)
        score, model = wrapper.train_and_score(
            candidate.params,
            X,
            y,
            scoring=scoring,
            custom_metric=custom_metric,
            task_type=task_type,
        )
        candidate.model = model
        candidate.score = score
    except Exception:
        candidate.score = float("-inf")
        candidate.model = None
        logging.exception("Evaluation failed for candidate %s", getattr(candidate, "params", None))
    return candidate


class ParallelExecutor:
    """Handles parallel evaluation of optimization candidates.

    Uses Python's multiprocessing pool to evaluate model candidates in parallel,
    improving throughput during model or hyperparameter search.
    """

    def __init__(self, num_workers=None):
        """Initialize the parallel executor.

        Args:
            num_workers (int, optional): Number of parallel workers to spawn.
                Defaults to `os.cpu_count() - 1` (all but one core).
        """
        self.num_workers = num_workers or max(1, os.cpu_count() - 1)

    def evaluate(
        self,
        candidates: List[Candidate],
        wrapper,
        X,
        y,
        scoring="accuracy",
        custom_metric=None,
        task_type="classification",
    ):
        """Evaluate multiple candidates in parallel.

        Args:
            candidates (List[Candidate]): List of candidate configurations to evaluate.
            wrapper: ModelWrapper instance providing `train_and_score()`.
            X (array-like): Training feature matrix.
            y (array-like): Target vector.
            scoring (str): Scoring metric name (e.g., `"accuracy"`).

        Returns:
            List[Candidate]: List of evaluated candidates with updated `score` and `model`.
        """

        # Prepare custom_metric for safe transmission to worker processes.
        # If the callable is not pickleable, try dill as a fallback.
        metric_obj = custom_metric

        if callable(custom_metric):
            serialization_success = False

            # Try pickle first
            try:
                pickle.dumps(custom_metric)
                metric_obj = custom_metric
                serialization_success = True
            except Exception:
                pass

            # Try dill if pickle failed
            if not serialization_success and dill is not None:
                try:
                    metric_obj = {
                        "__serialized__": True,
                        "data": dill.dumps(custom_metric),
                        "use_dill": True,
                    }
                    serialization_success = True
                except Exception:
                    pass

            # Last resort: still not serializable
            if not serialization_success:
                # If only one worker → fallback to sequential evaluation
                if self.num_workers == 1:
                    results = []
                    for cand in candidates:
                        try:
                            score, model = wrapper.train_and_score(
                                cand.params,
                                X,
                                y,
                                scoring=scoring,
                                custom_metric=custom_metric,
                                task_type=task_type,
                            )
                            cand.score = score
                            cand.model = model
                        except Exception:
                            cand.score = float("-inf")
                            cand.model = None
                        results.append(cand)
                    return results

                # Otherwise raise clear error
                raise RuntimeError(
                    "Provided custom_metric is not pickleable and dill is not available."
                )

        args = [
            (cand, wrapper, X, y, scoring, metric_obj, task_type)
            for cand in candidates
        ]
        with Pool(self.num_workers) as p:
            results = p.map(_eval_candidate_worker, args)
        return results
