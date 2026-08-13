# OptiFlowX examples

The examples are executable documentation. The `*_complete.py` files are designed to cover the public API of each course module rather than demonstrating only one theorem.

Run from the repository root after `python -m pip install -e .`:

```bash
python examples/01_markov_chain_complete.py
python examples/02_poisson_process_complete.py
python examples/03_cmtc_complete.py
python examples/04_birth_death_complete.py
python examples/05_conditional_expectation_complete.py
python examples/06_martingale_complete.py
python examples/07_theory_helpers_complete.py
```

Each complete example follows the same pattern:

1. construct the mathematical object;
2. inspect its state;
3. evaluate the main mathematical operations;
4. inspect structural properties and diagnostics;
5. simulate when the object supports simulation;
6. print small, reproducible results.

The examples intentionally use the same terminology as the Package / API pages while keeping the mathematical explanations in the Course material section.