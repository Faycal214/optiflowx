"""Public optimizer API exports.

Lazy re-exports to avoid circular imports with core.engine.
"""

__all__ = [
    "AntColonyOptimizer",
    "BayesianOptimizer",
    "GeneticOptimizer",
    "GreyWolfOptimizer",
    "PSOOptimizer",
    "RandomSearchOptimizer",
    "SimulatedAnnealingOptimizer",
    "TPEOptimizer",
]

def __getattr__(name):
    if name == "AntColonyOptimizer":
        from .ant_colony import AntColonyOptimizer
        return AntColonyOptimizer
    if name == "BayesianOptimizer":
        from .bayesian import BayesianOptimizer
        return BayesianOptimizer
    if name == "GeneticOptimizer":
        from .genetic import GeneticOptimizer
        return GeneticOptimizer
    if name == "GreyWolfOptimizer":
        from .grey_wolf import GreyWolfOptimizer
        return GreyWolfOptimizer
    if name == "PSOOptimizer":
        from .pso import PSOOptimizer
        return PSOOptimizer
    if name == "RandomSearchOptimizer":
        from .random_search import RandomSearchOptimizer
        return RandomSearchOptimizer
    if name == "SimulatedAnnealingOptimizer":
        from .simulated_annealing import SimulatedAnnealingOptimizer
        return SimulatedAnnealingOptimizer
    if name == "TPEOptimizer":
        from .tpe import TPEOptimizer
        return TPEOptimizer
    raise AttributeError(name)
