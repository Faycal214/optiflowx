from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from optiflowx.core import SearchSpace
from optiflowx.optimizers import GeneticOptimizer


def test_genetic_small():
    X, y = make_classification(n_samples=80, n_features=8, n_classes=2, random_state=0)

    search_space = SearchSpace()
    search_space.add("max_depth", "discrete", (2, 10))
    search_space.add("criterion", "categorical", ["gini", "entropy"])

    model_class = DecisionTreeClassifier

    optimizer = GeneticOptimizer(
        search_space,
        metric="accuracy",
        model_class=model_class,
        X=X,
        y=y,
        population=4,
        mutation_prob=0.2,
    )

    best_params, best_score = optimizer.run(max_iters=3)

    assert isinstance(best_params, dict)
    assert isinstance(best_score, float)
    assert 0 <= best_score <= 1
