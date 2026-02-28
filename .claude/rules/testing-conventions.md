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

## Tuning Tests (`test_<model>_scalar_tune_ml_models.py`)

Scalar models with `tune_ml_models()` require a dedicated test file. Add it alongside the 5 standard scalar test files.

### Fixtures and Shared Constants

```python
# Matches resolve_optuna_cv(cv=5) used internally — required for improvement assertions
_TUNE_CV = KFold(n_splits=5, shuffle=True, random_state=42)

@pytest.fixture(scope="module")
def <model>_data():
    np.random.seed(3141)
    return make_<model>_data(n_obs=500, dim_x=5)

@pytest.fixture(scope="module", params=["score_a", "score_b"])
def score(request):
    return request.param
```

### Required Tests

| Test | Checks |
|------|--------|
| `test_<model>_scalar_tune_basic` | Return type `dict[str, DMLOptunaResult]`; correct keys; `tuned=True`; params applied to learners; `model.fit()` succeeds. Parametrize over `score` + `_SAMPLER_CASES`. |
| `test_<model>_scalar_tune_improves_score` | `tune_res[name].best_score > cross_val_score(default_tree, ..., cv=_TUNE_CV, scoring="neg_root_mean_squared_error").mean()` |
| `test_<model>_scalar_tune_returns_self` | `return_tune_res=False` returns `self` |
| `test_<model>_scalar_tune_set_as_params_false` | Learner params unchanged; `best_params` still populated |
| `test_<model>_scalar_tune_invalid_key` | Unknown key raises `ValueError` |
| `test_<model>_scalar_tune_partial_space` | Tuning only a subset leaves unspecified learners unchanged |

For models with `_LEARNER_PARAM_ALIASES` (e.g., IRM `"ml_g"` → `["ml_g0", "ml_g1"]`), add:

| Test | Checks |
|------|--------|
| `test_<model>_scalar_tune_<alias>_alias` | Alias expands to concrete keys in result dict (not the alias key itself) |
| `test_<model>_scalar_tune_<alias>_alias_explicit_override` | Explicit concrete key overrides alias; verify by constraining the tuned range |

### Scalar vs. Old API

`DoubleMLScalar.tune_ml_models()` returns `dict[str, DMLOptunaResult]` **directly** — no repetition index. The old `DoubleML` API wraps results in a list (`tune_res[0]["ml_l"]`) because tuning runs per repetition. Scalar tuning uses the full dataset once, so the list dimension doesn't exist.

```python
# Scalar (new):   tune_res["ml_l"].best_params
# Old DoubleML:   tune_res[0]["ml_l"].best_params
```

---

## Naming

- Files: `test_<model>.py`, `test_<model>_scalar.py`, `test_<model>_scalar_exceptions.py`, `test_<model>_scalar_tune_ml_models.py`
- Functions: `test_<what>` — e.g., `test_coef_within_3_sigma`, `test_exception_invalid_score`
- Docstrings: Every test function gets a one-line docstring explaining what it verifies

## Checklist

- [ ] All tests marked `@pytest.mark.ci`
- [ ] Fixtures use `scope="module"` for model fitting
- [ ] Exception tests use `match=` with regex
- [ ] Seeds set for reproducibility
- [ ] Test functions have descriptive names and docstrings
- [ ] New scalar models have all 5 required test files (see `dml-scalar-test-structure.md`)
- [ ] If model has `tune_ml_models()`, add `test_<model>_scalar_tune_ml_models.py` with all required tuning tests
