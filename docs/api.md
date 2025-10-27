# API Reference

## Common interface
Each optimizer implements:

```python
optimizer = OptimizerClass(
    search_space=search_space,
    metric="accuracy",
    model_class=model_class,
    X=X,
    y=y,
    **params,
)
best_params, best_score = optimizer.run(max_iters=5)
```

## Key modules

```bash
optiflowx.core.search_space
```
Defines parameter search space and sampling.

```bash
optiflowx.models.configs
```

Contains model configuration classes for different estimators.

```bash
optiflowx.optimizers
```

Implements each optimization algorithm (GA, PSO, SA, etc.).

```bash
optiflowx.core.search_space
```

```bash
optiflowx.core.search_space
```
