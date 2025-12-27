"""
Example script to test multiple optimizers with a RandomForest model.
Make sure optiflowx is installed in editable mode:
    pip install -e .

Then run:
    python examples/test_all_optimizers.py
"""

from sklearn.datasets import make_classification
from optiflowx.optimizers import (
    GeneticOptimizer,
    PSOOptimizer,
    BayesianOptimizer,
    TPEOptimizer,
    RandomSearchOptimizer,
    SimulatedAnnealingOptimizer,
    GreyWolfOptimizer,
    AntColonyOptimizer,
)
from optiflowx.models.configs import RandomForestConfig, CustomModelConfig


"""
Example script to test multiple optimizers with a RandomForest model.
Make sure optiflowx is installed in editable mode:
    pip install -e .

Then run:
    python examples/test_all_optimizers.py
"""

from sklearn.datasets import make_classification
from optiflowx.optimizers import (
    GeneticOptimizer,
    PSOOptimizer,
    BayesianOptimizer,
    TPEOptimizer,
    RandomSearchOptimizer,
    SimulatedAnnealingOptimizer,
    GreyWolfOptimizer,
    AntColonyOptimizer,
)
from optiflowx.models.configs import RandomForestConfig, CustomModelConfig


# 1. Generate sample dataset
X, y = make_classification(
    n_samples=100,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    random_state=42,
)

use_custom = False  # Set to True to use the custom model example
if use_custom:
    """
    Example script to test multiple optimizers with a RandomForest model.
    Make sure optiflowx is installed in editable mode:
        pip install -e .

    Then run:
        python examples/test_all_optimizers.py
    """

    from sklearn.datasets import make_classification
    from optiflowx.optimizers import (
        GeneticOptimizer,
        PSOOptimizer,
        BayesianOptimizer,
        TPEOptimizer,
        RandomSearchOptimizer,
        SimulatedAnnealingOptimizer,
        GreyWolfOptimizer,
        AntColonyOptimizer,
    )
    from optiflowx.models.configs import RandomForestConfig, CustomModelConfig


    # 1. Generate sample dataset
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        random_state=42,
    )

    use_custom = False  # Set to True to use the custom model example
    if use_custom:
        """
        Example script to test multiple optimizers with a RandomForest model.
        Make sure optiflowx is installed in editable mode:
            pip install -e .

        Then run:
            python examples/test_all_optimizers.py
        """

        from sklearn.datasets import make_classification
        from optiflowx.optimizers import (
            GeneticOptimizer,
            PSOOptimizer,
            BayesianOptimizer,
            TPEOptimizer,
            RandomSearchOptimizer,
            SimulatedAnnealingOptimizer,
            GreyWolfOptimizer,
            AntColonyOptimizer,
        )
        from optiflowx.models.configs import RandomForestConfig, CustomModelConfig


        # 1. Generate sample dataset
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            random_state=42,
        )

        use_custom = False  # Set to True to use the custom model example
        if use_custom:
            cfg = CustomModelConfig()
        else:
            cfg = RandomForestConfig()
        search_space = cfg.build_search_space()
        model_class = cfg.get_wrapper().model_class

        # 3. Define optimizers to test
        optimizers = [
            ("pso", PSOOptimizer, {"n_particles": 10, "w": 0.7, "c1": 1.4, "c2": 1.4}),
            ("genetic", GeneticOptimizer, {"population": 10, "mutation_prob": 0.3}),
            ("bayesian", BayesianOptimizer, {"n_initial_points": 5}),
            ("tpe", TPEOptimizer, {"population_size": 10}),
            ("random_search", RandomSearchOptimizer, {"n_samples": 20}),
            (
                "simulated_annealing",
                SimulatedAnnealingOptimizer,
                {
                    "population_size": 10,
                    "initial_temp": 1.0,
                    "cooling_rate": 0.9,
                    "mutation_rate": 0.3,
                },
            ),
            ("grey_wolf", GreyWolfOptimizer, {"population_size": 10, "max_iters": 5}),
            ("ant_colony", AntColonyOptimizer, {"colony_size": 10, "max_iters": 5, "evaporation_rate": 0.2}),
        ]

        # 4. Run each optimizer and print results
        for opt_name, opt_class, opt_params in optimizers:
            print(f"\n{'='*30}\nTesting optimizer: {opt_name}\n{'='*30}")
            optimizer = opt_class(
                search_space=search_space,
                metric="accuracy",
                model_class=model_class,
                X=X,
                y=y,
                **opt_params,
            )

            best_params, best_score = optimizer.run()
            print(f"Result for {opt_name} → score={best_score:.4f}, params={best_params}")
