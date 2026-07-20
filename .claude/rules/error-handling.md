# Error Handling — DoubleML

> **Apply when**: Adding input validation, raising exceptions, or writing `pytest.raises` tests.

## Exception Type Mapping

| Situation | Exception | Example |
|-----------|-----------|---------|
| Invalid parameter value | `ValueError` | `n_folds < 2`, unknown `score` |
| Wrong argument type | `TypeError` | Non-`DoubleMLData` passed, class instead of instance |
| Property accessed before `fit()` | `ValueError` | `model.coef` before fitting |
| Wrong method call order | `ValueError` | `fit_nuisance_models()` before `draw_sample_splitting()` |

## Validation Patterns

### Use Project Helpers

Always use validation functions from `doubleml/utils/_checks.py`:

```python
from doubleml.utils._checks import _check_learner, _check_score, _check_finite_predictions

# Learner validation (checks sklearn compatibility, instance vs class)
self._learner_ml_l = _check_learner(ml_l, 'ml_l', regressor=True, classifier=False)

# Score validation
_check_score(score, valid_scores=['IV-type', 'partialling out'], allow_callable=True)
```

### Fail Fast — Validate in Constructor and Setters

```python
def __init__(self, data: DoubleMLData, score: str = "ATE") -> None:
    self._check_data(data)  # Validate immediately

    if score not in self._VALID_SCORES:
        raise ValueError(f"score must be one of {self._VALID_SCORES}, got '{score}'")
```

### Error Messages Must Include Expected vs. Actual

```python
# Good: specific and actionable
raise ValueError(f"n_folds must be at least 2, got {n_folds}")
raise TypeError(
    f"ml_m must be a classifier with predict_proba(). "
    f"Got {type(ml_m).__name__}. Did you pass a class instead of an instance?"
)

# Bad: vague
raise ValueError("Invalid input")
```

### Method Call Order Validation

```python
def fit_nuisance_models(self) -> None:
    if self._smpls is None:
        raise ValueError("Sample splitting has not been drawn. Call draw_sample_splitting() first.")

def estimate_causal_parameters(self) -> None:
    if self._predictions is None:
        raise ValueError("Nuisance models not fitted. Call fit_nuisance_models() first, or use fit().")
```

## Warnings vs. Exceptions

- **Exception**: Input is invalid, execution cannot continue
- **`warnings.warn()`**: Input is valid but may cause poor results

```python
# Warn on extreme propensity scores (valid but risky)
if np.any((propensity < 1e-12) | (propensity > 1 - 1e-12)):
    warnings.warn(
        f"Propensity scores close to 0 or 1 (eps=1e-12). "
        f"Trimming at {self._trimming_threshold}.",
        UserWarning
    )
```

## Testing Exceptions

Always use `match=` with regex to verify the error message:

```python
@pytest.mark.ci
def test_exception_invalid_score():
    msg = r"score must be one of .*, got 'invalid'"
    with pytest.raises(ValueError, match=msg):
        DoubleMLPLR(data, score='invalid')
```
