# Stochastic Processes User Guide

The stochastic-process part of StochX is the **course-faithful mathematical layer** of the package. It is intentionally separate from the time-series user guide, which is the main applied-analysis workflow.

The stochastic-process API covers the core objects used in applied probability courses:

- discrete-time Markov chains;
- Poisson processes;
- continuous-time Markov chains;
- birth-death processes;
- finite probability spaces and random variables;
- conditional expectation via partitions;
- filtrations and martingales;
- stopping times and stopped processes;
- trajectory analysis and simulation.

Each guide page uses the same pattern as the time-series guide: mathematical object → assumptions → Python representation → worked example → interpretation → API reference.

## Where to start

| Topic | Start here |
|---|---|
| DTMC | [Markov chains](markov-chains.md) |
| Poisson counting models | [Poisson processes](poisson-processes.md) |
| CTMC and birth-death models | [CTMC and birth-death](ctmc-birth-death.md) |
| Probability spaces and random variables | [Probability objects](probability-objects.md) |
| Filtration, martingale and stopping-time concepts | [Martingales](martingales.md) |
| Random trajectory generation | [Simulation](simulation.md) |

## Mathematical conventions

StochX validates matrices, probability vectors, state labels and model parameters at the public boundary. Numerical routines may use compact internal representations, but the public result preserves the semantic state labels and exposes validation failures explicitly.

## Relationship to the time-series package

The two parts share Python conventions and numerical discipline but serve different goals:

- **Stochastic processes:** mathematical objects and probabilistic computations.
- **Time series:** empirical data analysis, econometric modeling, diagnostics and forecasting.

Users coming from the USTHB probability/stochastic-process curriculum can use the stochastic pages as mathematical reference material, then switch to the time-series guide for applied modeling.
