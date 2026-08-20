# Filtrations, martingales and stopping times

## Filtration

A filtration $(\mathcal F_t)$ is an increasing family of information sets:

$$\mathcal F_s\subseteq\mathcal F_t, \qquad s\le t.$$

`Filtration` gives this abstract structure a concrete representation suitable for finite/discrete course exercises.

## Martingale

A process $(M_n)$ is a martingale with respect to $(\mathcal F_n)$ when

$$\mathbb E[|M_n|]<\infty, \qquad \mathbb E[M_{n+1}\mid\mathcal F_n]=M_n.$$

The StochX object is intended for executable course-style martingale checks and examples rather than for replacing a symbolic proof system.

## Stopping time

A stopping time $T$ satisfies

$$\{T\le n\}\in\mathcal F_n.$$

`StoppingTime` represents the stopping rule explicitly so that measurability conditions can be checked against the supplied filtration model.

## Stopped process

Given a process $X_n$ and a stopping time $T$, the stopped process is

$$X_{n\wedge T}.$$

```python
from stochx.stochastic import StoppedProcess

stopped = StoppedProcess(process, stopping_time)
```

## Practical study pattern

A useful course workflow is:

1. define the filtration;
2. define the process;
3. state the integrability condition;
4. check the martingale property;
5. define the stopping rule;
6. construct the stopped process;
7. study empirical or finite-sample behavior only after the mathematical assumptions are clear.

## API reference

See the `Filtration`, `Martingale`, `StoppingTime` and `StoppedProcess` reference pages for the concrete constructors and methods.
