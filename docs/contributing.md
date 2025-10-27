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
