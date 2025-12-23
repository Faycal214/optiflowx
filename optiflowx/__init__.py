"""
OptiFlow - lightweight hyperparameter optimization package.
Expose commonly used optimizers and registry here.
"""

__version__ = "0.0.7"

# lazy imports to avoid heavy deps on import
from .optimizers import *
from .models import registry as model_registry  # noqa: F401
