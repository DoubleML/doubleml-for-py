# Python Code Conventions — DoubleML

> **Apply when**: Writing or modifying any Python file in `doubleml/`.

## Tooling (from `pyproject.toml`)

| Tool | Config | Command |
|------|--------|---------|
| **black** | line-length=127, preview=true, target py310-313 | `black .` |
| **ruff** | rules E,F,W,I; ignores E721; target py312 | `ruff check .` / `ruff check --fix .` |
| **mypy** | `disallow_untyped_defs=true`, `no_implicit_optional=true`, excludes tests | `mypy doubleml` |
| **pre-commit** | black + ruff + trailing whitespace + debug statements | `pre-commit run --all-files` |

## File Structure

Every new or modified Python file must start with a **module-level docstring**. Do not add copyright headers, author/date stamps, or file paths — git tracks all of that.

### Module Docstring Patterns

Match the existing codebase style depending on file type:

```python
# Implementation files — one sentence: what the module contains
"""Partially Linear Regression (PLR) model based on the DoubleMLScalar hierarchy."""

# __init__.py files — Sphinx :mod: reference
"""The :mod:`doubleml.plm` module implements double machine learning estimates based on partially linear models."""

# Test files — one sentence: what is being tested
"""Compare PLR scalar against the existing DoubleMLPLR implementation."""
```

### Full File Header (implementation files)

```python
"""Partially Linear Regression (PLR) model based on the DoubleMLScalar hierarchy."""
from __future__ import annotations  # needed when class methods return Self/own type

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone

from doubleml.double_ml_scalar import DoubleMLScalar
from doubleml.utils._checks import _check_learner
```

Import order (enforced by ruff/isort): standard library, third-party, local.

### `from __future__ import annotations`

Not required in every file. Use it when a class references its own type in annotations (forward reference). Since the project targets Python 3.10+, `list[int]`, `dict[str, T]`, and `X | Y` unions work natively without it.

## Type Hints

All functions require complete type annotations including return types.

```python
def _nuisance_est(self, smpls: list[tuple[np.ndarray, np.ndarray]], n_rep: int = 1) -> dict[str, np.ndarray]:
```

- Use `-> None` for functions without return value
- Use `Optional[X]` or `X | None` (with `__future__` import) for nullable params
- `Any` is acceptable for scikit-learn estimators and dynamic learner objects
- Never suppress valid errors with `# type: ignore` — fix the type instead

## Docstrings (NumPy Style)

Required sections: **summary**, **Parameters**, **Returns**. Optional: **Raises**, **Examples**, **Notes**.

```python
def _get_score_elements(self, psi_predictions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Compute score elements from nuisance predictions.

    Parameters
    ----------
    psi_predictions : dict[str, np.ndarray]
        Dictionary with keys ``'ml_l'``, ``'ml_m'`` containing predictions of shape ``(n_obs,)``.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys ``'psi_a'`` (derivative) and ``'psi_b'`` (moment condition).
    """
```

Use `:class:\`~doubleml.DoubleMLData\`` for Sphinx cross-references. Use `.. math::` blocks for formulas.

## Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Modules | `snake_case` | `double_ml_plr.py` |
| Classes | `PascalCase` with `DoubleML` prefix | `DoubleMLPLR` |
| Methods/functions | `snake_case` | `fit_nuisance_models()` |
| Private methods | `_leading_underscore` | `_nuisance_est()` |
| Class variables | `_UPPER_SNAKE` | `_LEARNER_SPECS` |
| Constants | `UPPER_SNAKE` | `DEFAULT_N_FOLDS` |
| Statistical notation | Conventional names | `theta`, `se`, `psi_a`, `psi_b`, `n_obs`, `n_folds` |

## Class Design Patterns

### Property vs. Method

- **`@property`**: Computed attributes that are cheap and feel like data — `coef`, `se`, `summary`, `predictions`, `n_obs`, `n_folds`, `n_rep`, `score`
- **Methods**: Actions with side effects or expensive computation — `fit()`, `confint()`, `bootstrap()`, `draw_sample_splitting()`
- **`fit()` returns `self`** to enable chaining

### Class Variables vs. Instance Variables

- **Class variable**: Shared metadata — `_LEARNER_SPECS`, `_VALID_SCORES`
- **Instance variable**: Per-object state — `_dml_data`, `_smpls`, `_predictions`

### Decorators

- `@staticmethod` for stateless validation: `_check_data()`
- `@property` for computed attributes: `coef`, `se`
- `@abstractmethod` for template method hooks: `_nuisance_est()`, `_get_score_elements()`

### Score Function Contract

`_get_score_elements()` must return `dict[str, np.ndarray]` with:
- `'psi_a'`: Score derivative, shape `(n_obs,)`
- `'psi_b'`: Moment condition, shape `(n_obs,)`

Linear scores use closed-form: `theta = -mean(psi_b) / mean(psi_a)`.

## DoubleML-Specific Patterns

### Learner Handling

```python
# Always validate learners with _check_learner
self._learner_ml_l = _check_learner(ml_l, 'ml_l', regressor=True, classifier=False)

# Always clone before fitting (learners are mutable)
fitted_learner = clone(self._learner_ml_l).fit(X_train, y_train)

# Classifiers need predict_proba for propensity scores
propensity = fitted_learner.predict_proba(X_test)[:, 1]
```

### Sample Splitting

Always use `DoubleMLResampling`, never raw `KFold`:

```python
from doubleml.utils.resampling import DoubleMLResampling
resampling = DoubleMLResampling(n_folds=5, n_repeats=1, n_obs=n_obs)
```

### Vectorized Score Computation

```python
# Correct: vectorized NumPy operations
psi_a = -d_res * d_res          # shape: (n_obs,)
psi_b = d_res * (y - ml_g_hat)  # shape: (n_obs,)

# Wrong: Python loops over observations
```

Pre-allocate prediction arrays: `np.zeros((n_obs, n_rep))`.

### Error Messages

Include expected vs. actual values. Use `_check_*` helpers from `doubleml/utils/_checks.py`.

```python
if score not in self._VALID_SCORES:
    raise ValueError(f"score must be one of {self._VALID_SCORES}, got '{score}'")
```

Use `warnings.warn()` for non-fatal issues (e.g., extreme propensity scores), exceptions for invalid input.

## Verification Checklist

Before completing any task, run:

```bash
black .                    # Format
ruff check --fix .         # Lint + auto-fix
mypy doubleml              # Type check
pytest -m ci               # Tests
```

Check:
- [ ] All functions have type hints and return types
- [ ] File starts with a module-level docstring (one sentence, matching file type pattern)
- [ ] Public functions/classes have NumPy-style docstrings
- [ ] Learners validated with `_check_learner()`, cloned with `clone()` before fitting
- [ ] Score elements named `psi_a`/`psi_b`, shapes are `(n_obs,)`
- [ ] No `print()`, `breakpoint()`, or debug statements
- [ ] No magic numbers — use named constants
- [ ] Sample splitting uses `DoubleMLResampling`, not raw `KFold`
