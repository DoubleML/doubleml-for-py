# Testing Conventions — DoubleML

> **Apply when**: Writing or modifying test files in `doubleml/**/tests/`.

## Test Organization

```
doubleml/<module>/tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── test_<model>.py          # Legacy model tests
├── test_<model>_scalar.py   # Scalar model tests (see dml-scalar-test-structure.md)
└── ...
```

Package-level tests and utilities live in `doubleml/tests/` (with `_utils*.py` helpers).

## Markers

**All test functions must be marked `@pytest.mark.ci`** — this is the CI gate.

```python
@pytest.mark.ci
def test_coef_accuracy(fitted_model):
    ...

@pytest.mark.ci
@pytest.mark.parametrize("score", ["IV-type", "partialling out"])
def test_score_variants(score):
    ...
```

Other markers: `@pytest.mark.ci_rdd` for RDD-specific tests.

Run: `pytest -m ci` (CI), `pytest doubleml/plm/tests/` (module), `pytest -k "plr and scalar"` (pattern).

## Fixtures

### Use `scope="module"` for Expensive Operations

Model fitting is expensive. Fit once, share across tests:

```python
@pytest.fixture(scope="module")
def fitted_model():
    np.random.seed(42)
    data = make_plr_data(n_obs=200)
    dml_obj = DoubleMLPLRScalar(data, score="IV-type")
    dml_obj.set_learners(ml_l=Lasso(), ml_m=Lasso())
    dml_obj.draw_sample_splitting(n_folds=3, n_rep=2)
    dml_obj.fit()
    return dml_obj
```

### Parametrize for Multiple Scenarios

```python
@pytest.fixture(scope="module", params=["IV-type", "partialling out"])
def score(request):
    return request.param
```

Each combination creates one fixture instance shared across all tests in the module.

## Assertion Patterns

| Context | Pattern | Tolerance |
|---------|---------|-----------|
| Statistical accuracy | `abs(coef - true_theta) <= 3.0 * se` | 3-sigma rule |
| Backward compatibility | `np.testing.assert_allclose(new, old, rtol=1e-9)` | Exact match |
| External predictions | `math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-4)` | Small tolerance |
| Exception messages | `pytest.raises(ValueError, match=r"regex")` | Exact message |
| Types and shapes | `isinstance(x, np.ndarray)`, `x.shape == (n,)` | Exact |

### Key: Always Use `match=` for Exception Tests

```python
msg = r"score must be one of .*, got 'invalid'"
with pytest.raises(ValueError, match=msg):
    DoubleMLPLR(data, score='invalid')
```

## Reproducibility

- **Always seed**: `np.random.seed(42)` at the start of data generation
- **Share sample splits** in comparison tests: `dml_new._smpls = dml_old.smpls`
  (Old and new implementations consume random state differently during `__init__`)
- **Small data for speed**: `n_obs=200`, `n_folds=3` for return type / exception tests
- **Larger data for accuracy**: `n_obs=500`, `n_folds=5` for estimation tests

## Naming

- Files: `test_<model>.py`, `test_<model>_scalar.py`, `test_<model>_scalar_exceptions.py`
- Functions: `test_<what>` — e.g., `test_coef_within_3_sigma`, `test_exception_invalid_score`
- Docstrings: Every test function gets a one-line docstring explaining what it verifies

## Checklist

- [ ] All tests marked `@pytest.mark.ci`
- [ ] Fixtures use `scope="module"` for model fitting
- [ ] Exception tests use `match=` with regex
- [ ] Seeds set for reproducibility
- [ ] Test functions have descriptive names and docstrings
- [ ] New scalar models have all 5 required test files (see `dml-scalar-test-structure.md`)
