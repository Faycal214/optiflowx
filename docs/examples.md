# Worked Examples

The repository contains focused runnable examples for each stochastic-process area and a dedicated public-API coverage gallery.

| Example | Coverage |
|---|---|
| `examples/01_discrete_markov_chain.py` | Core `MarkovChain` construction and discrete-time operations |
| `examples/02_poisson_process.py` | Core `PoissonProcess` usage |
| `examples/02_poisson_complete.py` | Extended homogeneous and non-homogeneous Poisson operations |
| `examples/03_continuous_markov_chain.py` | Core `ContinuousTimeMarkovChain` usage |
| `examples/03_cmtc_complete.py` | Extended CTMC, jump-chain, holding-time, cost, and path operations |
| `examples/04_birth_death_process.py` | Birth-death construction and core formulas |
| `examples/05_conditional_expectation.py` | Core finite conditional-expectation usage |
| `examples/05_conditional_expectation_complete.py` | Extended probability-space, partition, random-variable, and conditional operations |
| `examples/06_martingale.py` | Core discrete-time martingale usage |
| `examples/api_quickstart.py` | Minimal public API entry point |
| `examples/07_api_operations.py` | Public classes and meaningful public operations across the full stochastic API |

The examples are executable documentation. CI runs the Python scripts and builds the documentation so that runnable examples remain synchronized with the public API.
