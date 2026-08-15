# Course → API map

This page maps the main course concepts to the corresponding OptiFlowX API components.

## Chapter 1 — Discrete-Time Markov Chains

| Course concept | OptiFlowX | Documentation |
|---|---|---|
| one-step transition matrix `P` | `MarkovChain` | `cmt_d.md` §2 |
| transition in `n` steps `P^(n)` | `n_step_transition` | `cmt_d.md` §3 |
| law `mu_n = mu_0 P^n` | `state_distribution` | `cmt_d.md` §4 |
| Chapman–Kolmogorov | `chapman_kolmogorov` | `cmt_d.md` §3 |
| accessibility | `accessible` | `cmt_d.md` §5 |
| communication | `communicate` | `cmt_d.md` §5 |
| communication classes | `communicating_classes` | `cmt_d.md` §5 |
| closed classes | `closed_classes` | `cmt_d.md` §5 |
| recurrent/transient classification | `classify_states` | `cmt_d.md` §6 |
| period | `period` | `cmt_d.md` §8 |
| stationary distribution | `stationary_distribution` | `cmt_d.md` §9 |
| limiting distribution | `limiting_distribution` | `cmt_d.md` §10 |
| absorption probability | `absorption_probability` | `cmt_d.md` §11 |

## Chapter 2 — Poisson Processes

| Course concept | OptiFlowX | Documentation |
|---|---|---|
| homogeneous Poisson process | `PoissonProcess` | `poisson.md` |
| Poisson counting law | `count_probability` | `poisson.md` |
| independent increments | increment utilities | `poisson.md` |
| interarrival times | `interarrival_samples` | `poisson.md` |
| arrival times | `arrival_times` | `poisson.md` |
| simulation | `simulate` | `poisson.md` |
| conditioning on counts | `conditional_first_arrival_cdf`, `conditional_arrival_times` | `poisson.md` |
| superposition | `superpose` | `poisson.md` |
| thinning | `split` | `poisson.md` |
| non-homogeneous process | `NonHomogeneousPoissonProcess` | `poisson.md` |

## Chapter 3 — Continuous-Time Markov Chains

| Course concept | OptiFlowX | Documentation |
|---|---|---|
| infinitesimal generator `Q` | `ContinuousTimeMarkovChain` | `cmtc.md` |
| transition matrix `P(t)` | `transition_matrix` | `cmtc.md` |
| infinitesimal transition approximation | `infinitesimal_transition_matrix` | `cmtc.md` |
| Chapman–Kolmogorov | `chapman_kolmogorov` | `cmtc.md` |
| Kolmogorov equations | `forward_derivative`, `backward_derivative` | `cmtc.md` |
| matrix exponential | `transition_matrix` | `cmtc.md` |
| stationary law | `stationary_distribution` | `cmtc.md` |
| holding rate / waiting time | `holding_rate`, `holding_time` | `cmtc.md` |
| embedded jump chain | `jump_chain_matrix`, `jump_chain` | `cmtc.md` |
| simulated trajectory | `CTMCPath`, `simulate` | `cmtc.md` |

## Birth-Death Processes

| Course concept | OptiFlowX | Documentation |
|---|---|---|
| birth/death rates | `BirthDeathProcess` | `birth_death.md` |
| finite generator | `generator_matrix` | `birth_death.md` |
| embedded jump chain | `jump_chain_matrix` | `birth_death.md` |
| Kolmogorov evolution | `kolmogorov_derivative` | `birth_death.md` |
| stationary product weights | `stationary_weights` | `birth_death.md` |
| pure birth/death/immigration examples | class constructors and formulas | `birth_death.md` |
| explosion criterion | `pure_birth_reciprocal_rate_sum` | `birth_death.md` |

## Chapter 4 — Conditional Expectation

| Course concept | OptiFlowX | Documentation |
|---|---|---|
| finite probability space | `FiniteProbabilitySpace` | `conditional_expectation.md` |
| random variable | `RandomVariable` | `conditional_expectation.md` |
| conditioning partition | `Partition` | `conditional_expectation.md` |
| `E(X | G)` | `conditional_expectation` | `conditional_expectation.md` |
| `E(X | Y)` | `conditional_expectation_given` | `conditional_expectation.md` |
| conditional probability | `conditional_probability` | `conditional_expectation.md` |
| total expectation | `total_expectation` | `conditional_expectation.md` |
| tower property | `tower` | `conditional_expectation.md` |
| conditional variance/covariance | `conditional_variance`, `conditional_covariance` | `conditional_expectation.md` |
| `L^2` projection | `l2_projection` | `conditional_expectation.md` |

## Chapter 5 — Discrete-Time Martingales

| Course concept | OptiFlowX | Documentation |
|---|---|---|
| filtration | `Filtration` | `martingales.md` |
| adapted process | `is_adapted` | `martingales.md` |
| martingale | `Martingale.is_martingale` | `martingales.md` |
| submartingale | `Martingale.is_submartingale` | `martingales.md` |
| supermartingale | `Martingale.is_supermartingale` | `martingales.md` |
| Doob martingale | `Martingale.doob` | `martingales.md` |
| stopping time | `StoppingTime` | `martingales.md` |
| stopped process | `StoppedProcess` | `martingales.md` |
