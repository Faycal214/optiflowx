# Course → API traceability

This page is the bridge between the five MSPRO PDFs and the OptiFlowX public API.

The purpose is not to reproduce the PDFs. It is to make the correspondence explicit: **where the course introduces a concept, which Python object represents it, and where the documentation explains it.**

## Chapter 1 — CMTD

**Source:** `Chapitre1_CMTD_2024-2025 (2)-1.pdf`

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

The course gives the stationary relation `pi_j = 1 / mu_j` in the irreducible positive-recurrent case and interprets `pi_j` as long-run time proportion. See Chapter 1, pp. 44–48. It treats the finite-chain closed-class result on p. 48. The limiting-distribution theorem appears on pp. 52–54, with the periodic counterexample on pp. 57–58.

## Chapter 2 — Poisson processes

**Source:** `Chapitre2_Processus de Poisson_2024-2025-1.pdf`

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

## Chapter 3 — CMTC

**Source:** `Chapitre3_CMTC_2024-2025-2.pdf`

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

The course defines stationarity by `pi P(t) = pi` and proves the finite-state criterion `pi Q = 0` together with normalization. See Chapter 3, pp. 27–29. The course introduces the holding-time description on p. 30.

## Chapter 3 — Birth-death processes

**Source:** `Chapitre3_CMTC_2024-2025-2.pdf`

| Course concept | OptiFlowX | Documentation |
|---|---|---|
| birth/death rates | `BirthDeathProcess` | `birth_death.md` |
| finite generator | `generator_matrix` | `birth_death.md` |
| embedded jump chain | `jump_chain_matrix` | `birth_death.md` |
| Kolmogorov evolution | `kolmogorov_derivative` | `birth_death.md` |
| stationary product weights | `stationary_weights` | `birth_death.md` |
| pure birth/death/immigration examples | class constructors and formulas | `birth_death.md` |
| explosion criterion | `pure_birth_reciprocal_rate_sum` | `birth_death.md` |

## Chapter 4 — Conditional expectation

**Source:** `Chapitre4_Esperance conditionne_2024-2025.pdf`

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

The PDF defines `E(X|Y)` through measurability with respect to `sigma(Y)` and equality of integrals on every event of `sigma(Y)`, then generalizes to an arbitrary sub-tribe. See Chapter 4, p. 23 and following.

## Chapter 5 — Discrete-time martingales

**Source:** `Chapitre5_Martingales_2024-2025-1.pdf`

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

The course defines a filtration as an increasing sequence of sub-tribes and defines adaptedness by measurability with respect to the corresponding filtration term. See Chapter 5, pp. 4–5.
