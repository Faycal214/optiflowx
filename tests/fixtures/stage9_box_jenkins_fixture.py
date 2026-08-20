"""Independent deterministic numerical fixture for Stage 9.7.

The sample generator intentionally duplicates the mathematical ARMA recursion
rather than importing a StochX simulator. This keeps the fixture independent
from the implementation under test.
"""

from __future__ import annotations

import numpy as np

SETTINGS = {
    "nobs": 180,
    "phi": (0.45,),
    "theta": (0.25,),
    "rng": 42,
    "burnin": 500,
    "alpha": 0.05,
    "validation_lags": 8,
    "criterion": "aic",
    "forecast_steps": 5,
}

EXPECTED = {
    "selected_order": (1, 0, 1),
    "llf": -260.8915765312392,
    "sigma_sq": 1.0,
    "aic": 529.7831530624784,
    "sc": 542.5549804660392,
    "hq": 534.9615791083713,
    "params": (-0.20007825, 0.25689138, 0.40527086, 1.06017370),
    "bse": (0.14510115, 0.12037751, 0.10470496, 0.11595724),
    "tvalues": (-1.37888813, 2.13404789, 3.87059848, 9.14279852),
    "pvalues": (0.16792925, 0.032838861, 0.000108568485, 6.08567043e-20),
    "ar_roots": (3.89269583,),
    "ma_roots": (-2.46748559,),
    "lb8_pvalue": 0.8763425395025081,
    "forecast": (0.53490052, -0.01126854, -0.15157466, -0.18761809, -0.19687734),
    "forecast_se": (1.02964737, 1.23491547, 1.24727409, 1.24808537, 1.24813889),
    "lower": (-1.48317124, -2.43165839, -2.59618696, -2.63382047, -2.64318462),
    "upper": (2.55297228, 2.40912131, 2.29303764, 2.25858428, 2.24942993),
}


def make_fixture_series() -> np.ndarray:
    """Return the frozen ARMA(1,1) sample used by Stage 9.7."""
    p = q = 1
    total = SETTINGS["nobs"] + max(SETTINGS["burnin"], p + q + 20)
    rng = np.random.default_rng(SETTINGS["rng"])
    eps = rng.normal(0.0, 1.0, total + q)
    x = np.zeros(total)
    for t in range(max(p, 1), total):
        ar_part = SETTINGS["phi"][0] * x[t - 1]
        ma_part = eps[t + q] + SETTINGS["theta"][0] * eps[t]
        x[t] = ar_part + ma_part
    return x[-SETTINGS["nobs"] :].copy()
