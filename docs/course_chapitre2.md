# Chapter 2 — Poisson Processes

## 1. Counting-process definition

A Poisson process $(N(t))_{t\ge0}$ with rate $\lambda>0$ starts at zero, has independent increments, stationary increments and

$$N(t)-N(s)\sim\operatorname{Poisson}(\lambda(t-s)), \qquad t>s.$$

Hence

$$\mathbb P(N(t)-N(s)=k)=e^{-\lambda(t-s)}\frac{[\lambda(t-s)]^k}{k!}.$$

## 2. Arrival times

Let $T_n$ be the time of the $n$-th event. The inter-arrival times are i.i.d.

$$T_n-T_{n-1}\sim\operatorname{Exp}(\lambda).$$

This gives two equivalent constructions of the process: simulate exponential waiting times or simulate counts over intervals.

## 3. Basic consequences

The increment over length $h$ has mean and variance

$$\mathbb E[N(t+h)-N(t)]=\lambda h,$$

$$\operatorname{Var}(N(t+h)-N(t))=\lambda h.$$

The equality of mean and variance is characteristic of the Poisson count model.

## 4. Conditional structure

Conditionally on the total number of arrivals in an interval, event times have the familiar order-statistics interpretation. This is useful for deriving arrival-time probabilities and simulation algorithms.

## 5. Superposition and thinning

Independent Poisson processes can be superposed to form a Poisson process with rate equal to the sum of the component rates. Conversely, thinning a Poisson process with constant retention probability gives another Poisson process with the scaled rate.

## 6. Non-homogeneous extension

If the intensity is time-dependent, $\lambda(t)$, the increment distribution is controlled by the integrated intensity

$$\Lambda(s,t)=\int_s^t\lambda(u)\,du.$$

Then

$$N(t)-N(s)\sim\operatorname{Poisson}(\Lambda(s,t)).$$

## 7. Simulation

Use an explicit generator in software experiments so that the same mathematical model produces reproducible paths.

## 8. StochX objects

`PoissonProcess` represents the homogeneous model and `NonHomogeneousPoissonProcess` represents the intensity-driven extension. Use them for event/arrival models; use the time-series namespace for regularly indexed measurements.

See the [Poisson-process guide](stochastic/poisson-processes.md) and API pages.
