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

## Evaluate Learners Tests (`test_<model>_scalar_evaluate_learners.py`)

Scalar models with `evaluate_learners()` require a dedicated test file. Constants: `N_OBS=500`, `N_FOLDS=5`, `N_REP=2`. Score-parametrized fixture (same pattern as tuning tests).

**Required tests:**

| Test | Checks |
|------|--------|
| `test_nuisance_loss_type_and_shape` | `dict`; each value `shape == (N_REP,)`; finite or NaN as expected |
| `test_nuisance_loss_positive` | RMSE > 0 for learners with real targets |
| `test_nuisance_targets_type_and_shape` | `shape == (N_OBS, N_REP)`; NaN arrays for unknown targets |
| `test_nuisance_targets_correct_values` | ml_l target == y; ml_m target == d (model-specific) |
| `test_evaluate_learners_default` | Default metric returns finite positive values |
| `test_evaluate_learners_rmse_matches_nuisance_loss` | `evaluate_learners(root_mean_squared_error)` equals `nuisance_loss` |
| `test_evaluate_learners_r2` | R² ≤ 1; correct shape |
| `test_evaluate_learners_mae` | MAE > 0; correct shape |
| `test_evaluate_learners_subset` | `learners=["ml_l"]` returns only `"ml_l"` key |
| `test_evaluate_learners_custom_metric` | Lambda metric matches sklearn equivalent |
| `test_evaluate_learners_before_fit_raises` | `ValueError` before `fit_nuisance_models()` |
| `test_evaluate_learners_after_reset_raises` | `ValueError` after `draw_sample_splitting()` |
| `test_nuisance_loss_before_fit_raises` | `ValueError` on `.nuisance_loss` before fit |
| `test_nuisance_targets_before_fit_raises` | `ValueError` on `.nuisance_targets` before fit |
| `test_evaluate_learners_invalid_learner` | Unknown learner name raises `ValueError` |
| `test_evaluate_learners_invalid_metric` | Non-callable metric raises `TypeError` |
| `test_reset_clears_nuisance` | After `draw_sample_splitting()`, `nuisance_loss` raises |

NaN conventions: PLR `ml_g` → all-NaN; IRM `ml_g0` → NaN for `d==1`; `ml_g1` → NaN for `d==0`.

---

## Sensitivity Tests (`test_<model>_scalar_sensitivity.py`)

Scalar models with `_sensitivity_element_est()` require a dedicated test file. Constants: `N_OBS=500`, `N_FOLDS=5`, `N_REP=2`. Score-parametrized `fitted_<model>` fixture.

**Exception tests** go in `test_<model>_scalar_exceptions.py` — not in this file:

| Test | Input | Expected |
|------|-------|----------|
| `test_exception_sensitivity_before_fit` | Call before `fit()` | `ValueError` matching `"The framework is not yet initialized"` |
| `test_exception_sensitivity_cf_y` | `cf_y=1` (int) / `cf_y=1.0` (boundary) | `TypeError` / `ValueError` |
| `test_exception_sensitivity_cf_d` | `cf_d=1` / `cf_d=1.0` | `TypeError` / `ValueError` |
| `test_exception_sensitivity_rho` | `rho=1` (int) / `rho=1.1` (out of range) | `TypeError` / `ValueError` |
| `test_exception_sensitivity_level` | `level=1` (int) / `level=0.0` (boundary) | `TypeError` / `ValueError` |
| `test_exception_sensitivity_null_hypothesis` | Wrong shape array | `ValueError` |

**Required tests** (parametrize over all scores):

| Test | Checks |
|------|--------|
| `test_sensitivity_elements_positive` | `sigma2 >= 0`, `nu2 > 0`, `max_bias >= 0` |
| `test_sensitivity_params_structure` | After `sensitivity_analysis()`: `theta/se/ci` have `lower`/`upper`; `rv`/`rva` in [0, 1] |
| `test_sensitivity_params_bounds_ordered` | `theta["lower"] <= coef <= theta["upper"]` |
| `test_sensitivity_rho0` | `rho=0.0`: `se["lower"] ≈ se["upper"] ≈ model.se` (`rtol=1e-6`) |
| `test_sensitivity_monotonicity_cf_y` | `cf_y=0.15` → wider theta bounds than `cf_y=0.03` |

---

## Naming

- Files: `test_<model>.py`, `test_<model>_scalar.py`, `test_<model>_scalar_exceptions.py`, `test_<model>_scalar_tune_ml_models.py`, `test_<model>_scalar_evaluate_learners.py`, `test_<model>_scalar_sensitivity.py`
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
- [ ] If model has `evaluate_learners()` / `nuisance_loss`, add `test_<model>_scalar_evaluate_learners.py`
- [ ] If model has `_sensitivity_element_est()`, add sensitivity exception tests to `test_<model>_scalar_exceptions.py` and add `test_<model>_scalar_sensitivity.py`
