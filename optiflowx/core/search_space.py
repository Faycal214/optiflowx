import random
import math
from typing import Dict, Any, List, Optional
import numpy as np


class SearchSpace:
    """Defines and samples hyperparameter search spaces."""

    def __init__(self, parameters: Optional[Dict[str, Dict[str, Any]]] = None):
        """Initialize a search space.

        Args:
            parameters (dict, optional): Dictionary defining parameters and their types.
                Example:
                {
                    "lr": {"type": "continuous", "values": (1e-4, 1e-1), "log": True},
                    "n_estimators": {"type": "discrete", "values": (50, 200)},
                    "criterion": {"type": "categorical", "values": ["gini", "entropy"]}
                }
        """
        self.parameters: Dict[str, Dict[str, Any]] = parameters or {}

    def add(self, name: str, param_type: str, values, log: bool = False):
        """Add a parameter to the search space.

        Args:
            name (str): Parameter name.
            param_type (str): One of `{"continuous", "discrete", "categorical"}`.
            values (tuple or list): Value range or list of options.
            log (bool, optional): Use logarithmic scale for continuous values.
                Defaults to False.
        """
        assert param_type in ("continuous", "discrete", "categorical")
        self.parameters[name] = {"type": param_type, "values": values, "log": log}

    def sample(self) -> Dict[str, Any]:
        """Sample a random hyperparameter configuration.

        Returns:
            dict: Randomly chosen parameter values within defined ranges.
        """
        out = {}
        for name, info in self.parameters.items():
            t, v, log = info["type"], info["values"], info["log"]

            if t == "continuous":
                low, high = v
                if log:
                    low = max(low, 1e-12)
                    out[name] = float(
                        math.exp(random.uniform(math.log(low), math.log(high)))
                    )
                else:
                    out[name] = float(random.uniform(low, high))

            elif t == "discrete":
                if (
                    isinstance(v, (list, tuple))
                    and len(v) == 2
                    and all(isinstance(x, (int, float)) for x in v)
                ):
                    low, high = int(v[0]), int(v[1])
                    out[name] = int(random.randint(low, high))
                else:
                    out[name] = random.choice(v)

            else:  # categorical
                out[name] = random.choice(v)

        return out

    def grid_sample(
        self, n_per_cont: int = 5, max_configs: Optional[int] = 10000
    ) -> List[Dict[str, Any]]:
        """Generate a grid of parameter combinations for exhaustive or hybrid search.

        Args:
            n_per_cont (int, optional): Number of samples per continuous dimension.
                Defaults to 5.
            max_configs (int, optional): Maximum allowed number of configurations.
                If exceeded, random subset is returned. Defaults to 10,000.

        Returns:
            list of dict: List of parameter dictionaries for grid search.
        """
        grids = [{}]
        for name, info in self.parameters.items():
            t, v, log = info["type"], info["values"], info["log"]
            new_grids = []

            if t == "categorical":
                choices = list(v)
                for g in grids:
                    for val in choices:
                        ng = g.copy()
                        ng[name] = val
                        new_grids.append(ng)

            elif t == "discrete":
                if (
                    isinstance(v, (list, tuple))
                    and len(v) == 2
                    and all(isinstance(x, (int, float)) for x in v)
                ):
                    low, high = int(v[0]), int(v[1])
                    vals = list(range(low, high + 1))
                else:
                    vals = list(v)
                for g in grids:
                    for val in vals:
                        ng = g.copy()
                        ng[name] = val
                        new_grids.append(ng)

            else:  # continuous
                low, high = v
                if log:
                    low = max(low, 1e-12)
                    vals = list(
                        np.exp(np.linspace(math.log(low), math.log(high), n_per_cont))
                    )
                else:
                    vals = list(np.linspace(low, high, n_per_cont))
                for g in grids:
                    for val in vals:
                        ng = g.copy()
                        ng[name] = float(val)
                        new_grids.append(ng)

            grids = new_grids

            if len(grids) > max_configs and max_configs is not None:
                return random.sample(grids, max_configs)

        if max_configs is not None and len(grids) > max_configs:
            return random.sample(grids, max_configs)

        return grids
