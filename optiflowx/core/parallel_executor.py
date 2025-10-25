from multiprocessing import Pool
from typing import List
from optiflowx.core.base_optimizer import Candidate
import os


def _eval_candidate_worker(args):
    """Worker function for parallel evaluation of a candidate.

    Args:
        args (tuple): Tuple containing `(candidate, wrapper, X, y, scoring)`.

    Returns:
        Candidate: The same candidate with updated `score` and optional fitted `model`.
    """
    candidate, wrapper, X, y, scoring = args
    try:
        score, model = wrapper.train_and_score(candidate.params, X, y, scoring=scoring)
        candidate.model = model
        candidate.score = score
    except Exception as e:
        candidate.score = float("-inf")
        candidate.model = None
        print(f"[WARN] Evaluation failed: {e}")
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

    def evaluate(self, candidates: List[Candidate], wrapper, X, y, scoring="accuracy"):
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
        args = [(cand, wrapper, X, y, scoring) for cand in candidates]
        with Pool(self.num_workers) as p:
            results = p.map(_eval_candidate_worker, args)
        return results
