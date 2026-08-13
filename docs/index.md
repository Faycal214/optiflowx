# OptiFlowX

## An executable companion to the MSPRO stochastic-process course

OptiFlowX is a Python library built around the five supplied USTHB MSPRO **Processus Aléatoires** course PDFs.

The project has one strict documentation rule:

> **First explain the mathematical object as it appears in the course. Then explain how OptiFlowX represents and computes that object.**

This makes the site both a learning resource and a professional scientific-library reference.

## The five course chapters

| Chapter | Subject | Main package objects |
|---|---|---|
| 1 | CMTD | `MarkovChain` |
| 2 | Processus de Poisson | `PoissonProcess`, `NonHomogeneousPoissonProcess` |
| 3 | CMTC | `ContinuousTimeMarkovChain`, `CTMCPath` |
| 3 | Naissance et mort | `BirthDeathProcess` |
| 4 | Espérance conditionnelle | `FiniteProbabilitySpace`, `RandomVariable`, `Partition` |
| 5 | Martingales à temps discret | `Filtration`, `Martingale`, `StoppingTime`, `StoppedProcess` |

## How each page is organized

Every mathematical topic follows the same structure:

1. **Definition from the course**
2. **Notation and formula**
3. **Theorem / proposition and hypotheses**
4. **Interpretation given by the course**
5. **OptiFlowX implementation**
6. **Worked example**
7. **Numerical limitations and scope**

For example, the stationary-distribution page does not start with `stationary_distribution()`. It first introduces the course definition

$$
\pi P=\pi,
$$

then explains the course's hypotheses and results, and only afterwards shows how the class computes the solution.

## Navigate by concept or by implementation

Use **Course material** when you are studying the mathematics.

Use **Library design** when you want to understand how the classes and numerical implementation are structured.

Use **Examples** when you want to run the concepts yourself.

## Source boundary

The five supplied PDFs are the primary mathematical source of the stochastic documentation. The implementation may use standard numerical tools such as NumPy and SciPy, but the educational pages do not silently replace a course definition with an unrelated external formulation.
