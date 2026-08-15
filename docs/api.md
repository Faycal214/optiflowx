# Package component reference

This page maps the public package components to the mathematical objects defined in the five course PDFs.

## Chapter 1 — CMTD

`MarkovChain` represents a finite homogeneous discrete-time Markov chain through its transition matrix $P$.

- `transition_matrix`: $P$
- `n_step_transition(n)`, `transition_matrix_at(n)`: $P^n$
- `state_distribution(mu0, n)`: $\mu_0P^n$
- `chapman_kolmogorov(m, n)`: $P^mP^n$
- `accessible`, `communicate`, `communicating_classes`: state classification by transitions
- `closed_classes`, `is_absorbing_state`: closed/absorbing classes
- `classify_states`: recurrence/transience
- `period`, `is_aperiodic`, `is_ergodic`: periodicity and ergodicity
- `first_visit_probability`, `first_passage_probability`: first hitting probabilities
- `first_return_probability`, `return_probability`, `mean_return_time`: return behavior
- `stationary_distribution`, `stationary_distributions`: stationary laws
- `limiting_distribution`: limit under the conditions represented in the course
- `absorption_probability`: probability of reaching a specified closed class
- `simulate`: trajectory generation

## Chapter 2 — Poisson process

`PoissonProcess(rate)` represents a homogeneous Poisson process.

- `count_probability`: $P(N(t)=n)$
- `increment_probability`: increment law
- `interarrival_samples`: exponential inter-arrival times
- `arrival_times`: cumulative arrival times
- `simulate`: event times up to a horizon
- `conditional_first_arrival_cdf`: conditional first-arrival law
- `conditional_arrival_times`: ordered arrival times conditional on the count
- `superpose`: sum of rates
- `split`: Bernoulli thinning/splitting

`NonHomogeneousPoissonProcess` represents the intensity $\lambda(t)$ and mean function

$$m(t)=\int_0^t\lambda(x)\,dx.$$

## Chapter 3 — CMTC

`ContinuousTimeMarkovChain` represents a finite homogeneous CTMC with generator $Q$.

- `generator_matrix`, `generator`: $Q$
- `infinitesimal_transition_matrix(h)`: $I+hQ$
- `transition_matrix(t)`, `transition_matrix_at(t)`: $e^{Qt}$
- `state_distribution`: $\mu_0P(t)$
- `chapman_kolmogorov`: $P(s+t)=P(s)P(t)$
- `forward_derivative`, `forward_equation`: $P(t)Q$
- `backward_derivative`, `backward_equation`: $QP(t)$
- `holding_rate`, `holding_time`: exponential holding time with rate $-q_{ii}$
- `jump_chain_matrix`, `jump_chain`: embedded jump chain
- `communicating_classes`: communication through the jump chain
- `stationary_distribution`, `stationary_distribution_from_jump_chain`: stationary laws
- `mean_return_time`: continuous-time return behavior
- `long_run_cost`: stationary weighted state cost
- `simulate`: jump-time trajectory

`CTMCPath` represents a simulated path.

- `state_at(t)`
- `occupation_time(state, horizon)`
- `occupation_fraction(state, horizon)`

## Birth-death process

`BirthDeathProcess` represents rates $\lambda_k$ and $\mu_k$.

- `birth_rate`, `death_rate`
- `generator`, `generator_matrix`
- `jump_chain`, `jump_chain_matrix`
- `kolmogorov_derivative`
- `stationary_weights`, `stationary_distribution`
- `pure_immigration_probability`
- `pure_birth_probability`
- `pure_death_probability`
- `pure_birth_reciprocal_rate_sum`

## Chapter 4 — Conditional expectation

`FiniteProbabilitySpace` contains a finite sample space and its probability law.

`RandomVariable` contains the values of a discrete random variable.

`Partition` represents the finite sigma-field/partition used for conditioning.

Main operations:

- `probability`, `probability_of`
- `conditional_probability_given_event`
- `conditional_expectation_given_event`
- `conditional_expectation`
- `conditional_expectation_given`
- `conditional_probability`
- `are_partitions_independent`, `are_independent`
- `conditional_characterization_error`
- `total_expectation`, `tower`, `pull_out`
- `conditional_variance`, `conditional_covariance`
- `total_variance`, `total_covariance`, `l2_projection`

## Chapter 5 — Martingales

`Filtration` represents an increasing sequence of finite partitions.

`Martingale` represents an adapted process and evaluates the conditional expectation defining a martingale, submartingale, or supermartingale.

- `is_martingale`
- `is_submartingale`
- `is_supermartingale`
- `martingale_residual`
- `conditional_future`
- `expectations`
- `transform`
- `doob`
- `stopped`

`StoppingTime` represents a discrete stopping time with values in $\mathbb N\cup\{\infty\}$.

`StoppedProcess` represents

$$X_n^T=X_{n\wedge T}.$$

- `values`
- `sequence`
- `terminal_value`

## Standalone analysis

`optiflowx.stochastic.analysis` contains generic trajectory analysis that does not belong to a single process object.

- `empirical_state_frequencies(path, states)`
