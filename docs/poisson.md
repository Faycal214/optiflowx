# Poisson Processes

This page gives a compact mathematical reference for homogeneous and non-homogeneous Poisson processes and shows how the concepts map to OptiFlowX.

## 1. Poisson process

Let \((N(t))_{t\ge0}\) be a counting process with rate \(\lambda>0\). The homogeneous Poisson model has independent increments, stationary increments, and

\[
P(N(t+s)-N(s)=n)=e^{-\lambda t}\frac{(\lambda t)^n}{n!}.
\]

```python
from optiflowx.stochastic import PoissonProcess

process = PoissonProcess(rate=2.0)
```

## 2. Counting law

On an interval of length \(t\),

\[
N(t)\sim\mathcal P(\lambda t),
\]

so

\[
P(N(t)=n)=e^{-\lambda t}\frac{(\lambda t)^n}{n!}.
\]

```python
process.count_probability(n=4, t=3.0)
```

For an interval \([s,t]\),

\[
N(t)-N(s)\sim\mathcal P(\lambda(t-s)).
\]

```python
process.increment_probability(n=2, s=1.0, t=4.0)
```

## 3. Inter-arrival and arrival times

The inter-arrival times are independent exponential random variables:

\[
T_n\sim\mathrm{Exp}(\lambda).
\]

```python
process.interarrival_samples(10)
process.arrival_times(10)
process.simulate(t_max=10.0)
```

## 4. Conditional arrival times

Conditionally on \(N(s)=1\),

\[
T_1\mid\{N(s)=1\}\sim\mathcal U([0,s]).
\]

Conditionally on \(N(s)=k\), the occurrence times have the distribution of uniform order statistics.

```python
process.conditional_first_arrival_cdf(y=1.5, s=3.0)
process.conditional_arrival_times(k=4, s=3.0)
```

## 5. Superposition and thinning

The superposition of independent Poisson processes with rates \(\lambda_1\) and \(\lambda_2\) is Poisson with rate

\[
\lambda_1+\lambda_2.
\]

```python
combined = PoissonProcess(2.0).superpose(PoissonProcess(3.0))
```

If each event is retained independently with probability \(p\), the two resulting processes have rates \(p\lambda\) and \((1-p)\lambda\).

```python
kept, rejected = process.split(0.3)
```

## 6. Non-homogeneous Poisson process

For a time-dependent intensity \(\lambda(t)\), define the cumulative mean

\[
m(t)=\int_0^t\lambda(u)\,du.
\]

```python
from optiflowx.stochastic import NonHomogeneousPoissonProcess

process = NonHomogeneousPoissonProcess(
    intensity=lambda t: 2 * t,
    mean_function=lambda t: t**2,
)

process.mean(3.0)
process.count_probability(2, 3.0)
```

## 7. API map

| Mathematical object | OptiFlowX |
|---|---|
| Homogeneous Poisson process | `PoissonProcess` |
| Counting law | `count_probability` |
| Independent increment law | `increment_probability` |
| Inter-arrival times | `interarrival_samples` |
| Arrival times | `arrival_times` |
| Simulation | `simulate` |
| Conditional first arrival | `conditional_first_arrival_cdf` |
| Conditional arrival times | `conditional_arrival_times` |
| Superposition | `superpose` |
| Thinning / splitting | `split` |
| Time-dependent intensity | `NonHomogeneousPoissonProcess` |
| Cumulative mean | `mean` |
