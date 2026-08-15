from pathlib import Path


EXAMPLES = "\n".join(
    path.read_text(encoding="utf-8")
    for path in Path("examples").glob("*.py")
)


PUBLIC_OBJECTS = {
    "MarkovChain": [
        "n_step_transition", "transition_matrix_at", "state_distribution",
        "chapman_kolmogorov", "transition_graph", "accessible", "communicate",
        "communicating_classes", "is_irreducible", "closed_classes",
        "is_absorbing_state", "classify_states", "first_visit_probability",
        "first_passage_probability", "visit_probability", "hitting_probability",
        "expected_hitting_time", "mean_hitting_time", "first_return_probability",
        "return_probability", "mean_return_time", "period", "is_aperiodic",
        "is_ergodic", "stationary_distribution", "stationary_distributions",
        "limiting_distribution", "absorption_probability", "simulate", "jump_chain",
    ],
    "PoissonProcess": [
        "count_probability", "increment_probability", "interarrival_samples",
        "arrival_times", "simulate", "count_sample", "count",
        "conditional_first_arrival_cdf", "conditional_arrival_times", "superpose", "split",
    ],
    "NonHomogeneousPoissonProcess": [
        "intensity_function", "mean", "count_probability", "increment_probability",
    ],
    "ContinuousTimeMarkovChain": [
        "infinitesimal_transition_matrix", "transition_matrix", "transition_matrix_at",
        "transition_matrix_uniformized", "transition_probability", "state_distribution",
        "chapman_kolmogorov", "forward_derivative", "forward_equation",
        "backward_derivative", "backward_equation", "stationary_distribution",
        "communicating_classes", "stationary_distribution_from_jump_chain",
        "mean_return_time", "long_run_cost", "holding_rate", "holding_time",
        "jump_chain_matrix", "jump_chain", "simulate",
    ],
    "CTMCPath": ["state_at", "occupation_time", "occupation_fraction"],
    "BirthDeathProcess": [
        "finite", "linear", "pure_immigration", "pure_birth", "pure_death",
        "birth_rate", "death_rate", "generator_matrix", "to_ctmc", "jump_chain_matrix",
        "jump_chain", "kolmogorov_derivative", "stationary_weights",
        "stationary_weights_at", "stationary_distribution", "pure_immigration_probability",
        "pure_birth_probability", "pure_death_probability", "pure_birth_reciprocal_rate_sum",
    ],
    "FiniteProbabilitySpace": [
        "probability", "probability_of", "random_variable", "partition",
        "conditional_probability_given_event", "conditional_expectation_given_event",
        "conditional_expectation", "conditional_expectation_given", "conditional_probability",
        "are_partitions_independent", "are_independent", "conditional_characterization_error",
        "total_expectation", "tower", "pull_out", "conditional_variance",
        "conditional_covariance", "total_variance", "total_covariance", "l2_projection",
    ],
    "RandomVariable": ["array", "expectation", "expected_value", "transform", "apply"],
    "Partition": ["from_blocks", "generated_by", "contains", "refines"],
    "Filtration": ["natural", "at", "is_adapted"],
    "Martingale": [
        "value_at", "conditional_next", "martingale_residual", "is_martingale",
        "is_submartingale", "is_supermartingale", "conditional_future", "expectations",
        "transform", "doob", "stopped",
    ],
    "StoppingTime": ["from_values", "minimum", "maximum", "add"],
    "StoppedProcess": ["values", "sequence", "terminal_value"],
}


def test_public_class_names_are_demonstrated() -> None:
    for class_name in PUBLIC_OBJECTS:
        assert class_name in EXAMPLES, f"No executable example mentions {class_name}"


def test_meaningful_public_operations_are_demonstrated() -> None:
    missing = [
        f"{class_name}.{operation}"
        for class_name, operations in PUBLIC_OBJECTS.items()
        for operation in operations
        if operation not in EXAMPLES
    ]
    assert not missing, "Missing example coverage: " + ", ".join(missing)


def test_examples_are_present_for_all_course_areas() -> None:
    required = (
        "01_discrete_markov_chain.py",
        "02_poisson_process.py",
        "03_continuous_markov_chain.py",
        "04_birth_death_process.py",
        "05_conditional_expectation.py",
        "06_martingale.py",
        "07_api_operations.py",
    )
    for filename in required:
        assert Path("examples", filename).exists(), f"Missing runnable example: {filename}"
