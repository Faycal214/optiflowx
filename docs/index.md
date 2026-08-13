# OptiFlowX mathematical reference

OptiFlowX is the computational implementation of the five supplied USTHB MSPRO **Processus Aléatoires** PDFs. The documentation follows the same mathematical scope.

## Source boundary

The package contains only:

1. Chaînes de Markov à temps discret (CMTD).
2. Processus de Poisson.
3. Chaînes de Markov à temps continu (CMTC).
4. Processus de naissance et de mort, as presented in the CMTC chapter.
5. Espérance conditionnelle.
6. Martingales à temps discret.

The implementation is intentionally finite/discrete when the corresponding course chapter uses finite or discrete objects. Numerical routines expose the formulas from the course; they are not intended as a replacement for a general symbolic probability system.

## How to read the package

Each mathematical component is documented in four layers:

- **Mathematical object:** the definition used in the PDF.
- **Result:** the proposition/theorem or formula used in the course.
- **Computation:** the corresponding OptiFlowX object or function.
- **Example:** a worked Python example under `examples/`.

## Examples

The repository contains one worked script per course chapter:

- `examples/01_discrete_markov_chain.py`
- `examples/02_poisson_process.py`
- `examples/03_continuous_markov_chain.py`
- `examples/04_birth_death_process.py`
- `examples/05_conditional_expectation.py`
- `examples/06_martingale.py`

## Source discipline

The mathematical explanations in this documentation are deliberately restricted to the definitions, propositions, theorems, examples, and constructions appearing in the five supplied PDFs.