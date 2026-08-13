# Package component reference

This page maps the public package components to the mathematical objects defined in the five course PDFs.

## Chapter 1 — CMTD

`MarkovChain` represents a finite homogeneous discrete-time Markov chain through its transition matrix $P$.

- `transition_matrix`: $P$
- `n_step_transition(n)`: $P^n$
- `state_distribution(mu0, n)`: $\mu_0P^n$
- `chapman_kolmogorov(m, n)`: $P^mP^n$
- `accessible`, `communicate`, `communicating_classes`: state classification by transitions
- `closed_classes`, `is_absorbing_state`: closed/absorbing classes
- `classify_states`: recurrence/transience
- `period`, `is_aperiodic`, `is_ergodic`: periodicity and ergodicity
- `first_visit_probability`: $P_i(T_j=n)$ for $i\ne j$
- `stationary_distribution`: irreducible finite stationary law
- `limiting_distribution`: limit under the conditions represented in the course
- `absorption_probability`: probability of reaching a specified closed class
- `simulate`: trajectory generation

Additional Chapter 1 helpers are in `theory.py`:

- `first_return_probability`
- `return_probability`
- `mean_return_time`
- `stationary_distributions`
- `empirical_state_frequencies`

## Chapter 2 — Poisson process

`PoissonProcess(rate)` represents a homogeneous Poisson process.

- `count_probability`: $P(N(t)=n)$
- `increment_probability`: increment law
- `interarrival_samples`: exponential inter-arrival times
- `arrival_times`: cumulative arrival times
- `simulate`: event times up to a horizon
- `conditional_first_arrival_cdf`: conditional uniform law of the first arrival
- `conditional_arrival_times`: ordered uniform arrival times conditional on the count
- `superpose`: sum of rates
- `split`: Bernoulli thinning/splitting

`NonHomogeneousPoissonProcess` represents the intensity $\lambda(t)$ and mean function

$$m(t)=\int_0^t\lambda(x)\,dx.$$

## Chapter 3 — CMTC

`ContinuousTimeMarkovChain` represents a finite homogeneous CTMC with generator $Q$.

- `generator_matrix`: $Q$
- `infinitesimal_transition_matrix(h)`: $I+hQ$
- `transition_matrix(t)`: $e^{Qt}$
- `state_distribution`: $\mu_0P(t)$
- `chapman_kolmogorov`: $P(s+t)=P(s)P(t)$
- `forward_derivative`: $P(t)Q$
- `backward_derivative`: $QP(t)$
- `holding_rate`, `holding_time`: exponential holding time with rate $-q_{ii}$
- `jump_chain_matrix`, `jump_chain`: embedded jump chain
- `stationary_distribution`: solution of $\pi Q=0$
- `long_run_cost`: stationary weighted state cost
- `simulate`: jump-time trajectory

Additional Chapter 3 helpers:

- `ctmc_communication_classes`
- `ctmc_stationary_from_jump_chain`
- `ctmc_mean_return_time`
- `occupation_time`
- `occupation_fraction`

## Birth-death process

`BirthDeathProcess` represents rates $\lambda_k$ and $\mu_k$.

- `birth_rate`, `death_rate`
- `generator_matrix`
- `jump_chain_matrix`
- `kolmogorov_derivative`
- `stationary_weights`
- `pure_immigration_probability`
- `pure_birth_probability`
- `pure_death_probability`
- `pure_birth_reciprocal_rate_sum`

## Chapter 4 — Conditional expectation

`FiniteProbabilitySpace` contains a finite sample space and its probability law.

`RandomVariable` contains the values of a discrete random variable.

`Partition` represents the finite sigma-field/partition used for conditioning.

Main operations:

- `expectation`
- `conditional_expectation`
- `conditional_expectation_given`
- `conditional_probability`
- `total_expectation`
- `tower`
- `pull_out`
- `conditional_variance`
- `conditional_covariance`
- `total_variance`
- `total_covariance`
- `l2_projection`

Additional helpers:

- `conditional_expectation_given_event`
- `conditional_probability_given_event`
- `independent_partitions`
- `independent_random_variables`
- `conditional_characterization_error`

## Chapter 5 — Martingales

`Filtration` represents an increasing sequence of finite partitions.

`Martingale` represents an adapted process and evaluates the conditional expectation defining a martingale, submartingale, or supermartingale.

- `is_martingale`
- `is_submartingale`
- `is_supermartingale`
- `martingale_residual`
- `conditional_future`
- `expectations`
- `doob`

`StoppingTime` represents a discrete stopping time with values in $\mathbb N\cup\{\infty\}$.

- `minimum`
- `maximum`
- `add`

`StoppedProcess` represents

$$X_n^T=X_{n\wedge T}.$$

- `values`
- `sequence`
- `terminal_value`

Additional helpers:

- `transform_martingale`
- `stopped_martingale`
