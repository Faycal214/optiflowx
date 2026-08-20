# Course Material

This section is the mathematical reference behind StochX. It is deliberately separate from the Python API: equations, definitions, hypotheses and decision rules are explained here first; implementation details live in the API and User Guide sections.

## Track A — Time Series and Econometrics

This is the primary applied track for users coming from EViews or a university time-series course.

### Workflow

```text
observed series
→ descriptive analysis
→ transformations
→ stationarity / deterministic specification
→ ACF / PACF
→ model identification
→ estimation
→ residual validation
→ selection
→ forecast
```

The software workflow mirrors this sequence through `Workfile`, expressions, ADF/DF/KPSS/PP tests, correlograms, AR/MA/ARMA/ARIMA/SARIMA estimators, diagnostics, Box–Jenkins helpers and forecast objects.

The [Time Series User Guide](time-series/index.md) is the practical entry point; [Time Series](time_series.md) contains the longer course-oriented workflow notes.

### Course topics to expand here

- weak stationarity and unit roots;
- deterministic components and trend specifications;
- lag operators and differencing;
- ACF/PACF interpretation;
- AR, MA and ARMA equations;
- ARIMA and seasonal ARIMA construction;
- information criteria: AIC, SC/BIC and HQ;
- residual whiteness and model adequacy;
- prediction and interval forecasting;
- linear state-space representation;
- Kalman filtering, smoothing and innovations.

## Track B — Stochastic Processes

### Chapter 1 — Discrete-Time Markov Chains

Transition matrices, n-step transition probabilities, Chapman–Kolmogorov, accessibility, communication classes, recurrence and transience, periodicity, irreducibility, stationary distributions, limiting behavior, hitting/return times and simulation.

[Chapter 1 — DTMC](course_chapitre1.md) · [Markov-chain guide](stochastic/markov-chains.md)

### Chapter 2 — Poisson Processes

Counting processes, independent and stationary increments, exponential inter-arrival times, event-count probabilities, conditional properties and non-homogeneous intensities.

[Chapter 2 — Poisson processes](course_chapitre2.md) · [Poisson guide](stochastic/poisson-processes.md)

### Chapter 3 — Continuous-Time Markov Chains

Generator matrices, holding times, transition semigroups, matrix exponentials, Chapman–Kolmogorov in continuous time, trajectories and the birth-death specialization.

[Chapter 3 — CTMC](course_chapitre3.md) · [CTMC guide](stochastic/ctmc-birth-death.md)

### Chapter 4 — Conditional Expectation

Finite probability spaces, random variables, partitions, sigma-algebra intuition, conditional expectation on partition cells and the tower-property viewpoint.

[Chapter 4 — Conditional expectation](course_chapitre4.md) · [Probability objects guide](stochastic/probability-objects.md)

### Chapter 5 — Discrete-Time Martingales

Filtrations, adapted processes, martingales, integrability, stopping times, stopped processes and finite-sample checks.

[Chapter 5 — Martingales](course_chapitre5.md) · [Martingale guide](stochastic/martingales.md)

## Track C — Reproducible simulation

Simulation is the computational bridge between theory and testing. Every generator should be paired with theoretical moments, path properties or deterministic fixtures where possible.

[Simulation guide](stochastic/simulation.md)

## Documentation philosophy

The course section answers **why the mathematics works**. The User Guide answers **when and how to use the method**. The API Reference answers **what the Python object accepts and returns**. Worked examples show the complete workflow.
