# Contributing to OptiFlowX

Thanks for your interest in contributing! This document describes how to propose changes, add new optimizers, run tests, and keep the documentation up to date.

## Quick start

1. Fork the repository and create a descriptive branch for your work:

```bash
git clone https://github.com/Faycal214/optiflowx.git
cd optiflowx
git checkout -b feat/short-description
```

2. Keep changes focused and add tests for any new behavior.

## Project layout (top-level)

- `optiflowx/` — Python package (core modules)
- `optiflowx/optimizers/` — optimizer implementations (GA, PSO, TPE, etc.)
- `optiflowx/models/configs/` — model configs and wrappers
- `examples/` — runnable examples for common workflows
- `docs/` — MkDocs documentation (source used for site)
- `tests/` — unit and integration tests

## Coding guidelines

- Use `black` for formatting and `ruff` for linting. These are used in CI.
- Add clear docstrings for new public APIs.
- Keep commits small and focused; update docs and examples when public behavior changes.

## Adding a new optimizer

1. Create a new file under `optiflowx/optimizers/` (for example `my_optimizer.py`).
2. Implement a class that accepts at minimum: `search_space`, `metric` or `custom_metric`, `model_class`, `X`, `y` and exposes a `run(max_iters=...)` method returning `(best_params, best_score)`.
3. Provide a short example in `examples/` and add unit tests under `tests/` demonstrating basic usage (use small sklearn datasets).

Example skeleton:

```python
class MyOptimizer:
    def __init__(self, search_space, metric, model_class, X, y, **kwargs):
        ...

    def run(self, max_iters=10):
        return best_params, best_score
```

## Documentation

- Update or add docs under `docs/` when you introduce new features.
- The documentation site is built with Docusaurus and lives in the `website/` folder.

To preview the documentation locally (from the repository root):

```bash
cd website
npm install
npm run start
```

To build a production-ready static site:

```bash
cd website
npm run build
```

## Tests

Install test dependencies and run the suite:

```bash
pip install -r requirements-test.txt
pytest -q
```

- Aim to write deterministic tests. If randomness is involved, set seeds in tests.

## Pull request checklist

- [ ] Branch from `main` and rebase/merge latest changes before opening a PR.
- [ ] Add or update tests for new behaviors.
- [ ] Update documentation and examples when the public API or semantics change.
- [ ] Run `black` and `ruff` locally and ensure no linting errors.
- [ ] Provide a clear PR description and link related issues.

If you are unsure about design, open an issue to discuss before implementing large changes. Maintainers are happy to help scope features and offer suggestions.

Thank you for improving OptiFlowX!
# Contributing to OptiFlowX

## Folder structure

optiflowx/
core/
optimizers/
models/
configs/
wrappers/
tests/
docs/

## Adding a new optimizer

1. Create a new file in `optiflowx/optimizers/`.
2. Follow the structure of existing optimizers (`genetic.py`, `pso.py`, etc.).
3. Implement a `.run(max_iters)` method returning `(best_params, best_score)`.

## Testing

Use pytest:

```bash
pytest -q
