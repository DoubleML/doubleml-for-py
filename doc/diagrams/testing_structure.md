# Testing Structure for DoubleML Scalar Models

This document defines the testing standard for all new models built on the `DoubleMLScalar` hierarchy. Each model should have a consistent set of test files covering estimation accuracy, return types, input validation, backward compatibility, and external predictions.

## Test File Convention

For a model `<model>` in module `<module>/` (e.g., `plr` in `plm/`, `irm` in `irm/`):

| File | Purpose |
|------|---------|
| `test_<model>_scalar.py` | Core estimation accuracy |
| `test_<model>_scalar_return_types.py` | Property types and shapes after fitting |
| `test_<model>_scalar_exceptions.py` | Input validation and error handling |
| `test_<model>_scalar_vs_<model>.py` | Comparison with old `DoubleML` implementation |
| `test_<model>_scalar_external_predictions.py` | External predictions workflow |

All test files live in `doubleml/<module>/tests/`.

All test functions should be marked with `@pytest.mark.ci`.

---

## 1. Core Estimation Tests (`test_<model>_scalar.py`)

Verify that the model produces statistically reasonable estimates.

### Fixture Pattern

```python
@pytest.fixture(scope="module", params=[...])  # score variants
def score(request):
    return request.param

@pytest.fixture(scope="module", params=[True, False])  # model-specific options
def option(request):
    return request.param

@pytest.fixture(scope="module")
def fitted_fixture(score, option):
    np.random.seed(3141)
    data = make_<model>_data(theta=true_theta, n_obs=500, ...)
    dml_obj = <Model>(data, score=score, option=option)
    dml_obj.set_learners(...)
    dml_obj.draw_sample_splitting(n_folds=5, n_rep=1)
    dml_obj.fit()
    return {"coef": dml_obj.coef[0], "se": dml_obj.se[0], "true_theta": true_theta, "score": score}
```

### Required Tests

- **`test_coef`**: For scores where the DGP theta equals the target parameter, check the 3-sigma rule: `abs(coef - true_theta) <= 3.0 * se`. For scores where the true parameter differs from the DGP theta (e.g., ATTE), check `np.isfinite(coef)` and `abs(coef) < 10.0`.
- **`test_se`**: `se > 0`

### Assertion Pattern

```python
# When true parameter matches DGP theta
assert abs(coef - true_theta) <= 3.0 * se

# When true parameter is unknown (e.g., ATTE with heterogeneous effects)
assert np.isfinite(coef)
assert abs(coef) < 10.0
```

---

## 2. Return Types Tests (`test_<model>_scalar_return_types.py`)

Verify that all properties have the correct types and shapes after fitting.

### Constants

```python
N_OBS = 200  # small for speed
N_FOLDS = 3
N_REP = 2
```

### Fixture Pattern

```python
@pytest.fixture(scope="module")
def fitted_model():
    np.random.seed(42)
    data = make_<model>_data(n_obs=N_OBS, ...)
    dml_obj = <Model>(data, score=<default_score>)
    dml_obj.set_learners(...)
    dml_obj.draw_sample_splitting(n_folds=N_FOLDS, n_rep=N_REP)
    dml_obj.fit()
    return dml_obj
```

### Required Tests

| Test | Assertion |
|------|-----------|
| `test_coef_type_and_shape` | `isinstance(coef, np.ndarray)`, `shape == (1,)` |
| `test_se_type_and_shape` | `isinstance(se, np.ndarray)`, `shape == (1,)` |
| `test_all_thetas_shape` | `shape == (1, N_REP)` |
| `test_all_ses_shape` | `shape == (1, N_REP)` |
| `test_summary_type` | `isinstance(summary, pd.DataFrame)`, `len == 1` |
| `test_confint_type_and_shape` | `isinstance(ci, pd.DataFrame)`, `shape == (1, 2)` |
| `test_psi_shape` | `shape == (N_OBS, 1, N_REP)` |
| `test_predictions_type` | `isinstance(predictions, dict)`, correct keys, each `shape == (N_OBS, N_REP)` |
| `test_smpls_type` | `len(smpls) == N_REP`, each has `N_FOLDS` tuples of `(train, test)` arrays |
| `test_n_properties` | `n_obs == N_OBS`, `n_folds == N_FOLDS`, `n_rep == N_REP`, `score == <expected>` |
| `test_required_learners` | Returns expected list of learner names |
| `test_str_repr` | `str(model)` and `repr(model)` return `str` |
| `test_get_params` | Returns dict with expected learner keys |
| `test_set_params` | Modifies and confirms learner parameter change |
| `test_before_fit_raises` | Accessing `coef` / `se` before `fit()` raises appropriate error |

---

## 3. Exception Tests (`test_<model>_scalar_exceptions.py`)

Verify that invalid inputs produce clear error messages.

### Required Tests (Common to All Models)

| Test | Input | Expected |
|------|-------|----------|
| `test_exception_data` | Non-DoubleMLData | `TypeError` |
| `test_exception_score` | Invalid score string | `ValueError` |
| `test_exception_n_folds` | `n_folds < 2` | `ValueError` |
| `test_exception_n_rep` | `n_rep < 1` | `ValueError` |
| `test_exception_fit_nuisance_without_smpls` | Call `fit_nuisance_models()` before `draw_sample_splitting()` | `ValueError` |
| `test_exception_estimate_causal_without_predictions` | Call `estimate_causal_parameters()` before `fit_nuisance_models()` | `ValueError` |
| `test_exception_missing_learner` | Call `fit()` without setting required learners | `ValueError` |
| `test_exception_invalid_learner` | Pass a class instead of an instance | `TypeError` |

### Model-Specific Exception Tests

Add tests for model-specific constraints:
- **PLR**: Instrumental variables check (`z_cols`), `ml_g` warning for partialling out
- **IRM**: Binary treatment check, instruments check, `normalize_ipw` type check, `ml_m` must be classifier

### Assertion Pattern

```python
@pytest.mark.ci
def test_exception_data():
    msg = r"The data must be of DoubleMLData type\."
    with pytest.raises(TypeError, match=msg):
        <Model>(pd.DataFrame())
```

Always use `match=` with regex patterns to verify error messages.

---

## 4. Comparison Tests (`test_<model>_scalar_vs_<model>.py`)

Verify exact numerical equivalence with the old `DoubleML` implementation.

### Fixture Pattern

```python
@pytest.fixture(scope="module", params=[...])
def score(request):
    return request.param

@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param

@pytest.fixture(scope="module")
def comparison_fixture(score, n_rep):
    n_folds = 5
    seed = 3141

    np.random.seed(42)
    data = make_<model>_data(...)

    # Old model
    np.random.seed(seed)
    dml_old = dml.DoubleML<Model>(data, learner1, learner2, n_folds=n_folds, n_rep=n_rep, score=score)
    dml_old.fit()

    # New model — share sample splits from old model
    dml_new = <Model>(data, score=score)
    dml_new.set_learners(...)
    dml_new._n_folds = n_folds
    dml_new._n_rep = n_rep
    dml_new._smpls = dml_old.smpls
    dml_new.fit()

    return {"old": dml_old, "new": dml_new}
```

**Key**: Share sample splits from the old model directly (`dml_new._smpls = dml_old.smpls`) because the old and new implementations consume random state differently during `__init__`.

### Required Tests

```python
def test_coef_equal(comparison_fixture):
    np.testing.assert_allclose(new.coef, old.coef, rtol=1e-9)

def test_se_equal(comparison_fixture):
    np.testing.assert_allclose(new.se, old.se, rtol=1e-9)

def test_all_coef_equal(comparison_fixture):
    np.testing.assert_allclose(new.all_thetas, old.all_coef, rtol=1e-9)

def test_all_se_equal(comparison_fixture):
    np.testing.assert_allclose(new.all_ses, old.all_se, rtol=1e-9)
```

Note the property name differences: new uses `all_thetas`/`all_ses`, old uses `all_coef`/`all_se`.

---

## 5. External Predictions Tests (`test_<model>_scalar_external_predictions.py`)

Verify that providing pre-computed predictions produces equivalent results.

### Fixture Pattern

```python
@pytest.fixture(scope="module", params=[...])
def score(request):
    return request.param

@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param

@pytest.fixture(scope="module", params=[True, False])
def set_ml_x_ext(request):   # one fixture per learner
    return request.param

@pytest.fixture(scope="module")
def ext_pred_fixture(score, n_rep, set_ml_x_ext, ...):
    # 1. Fit reference model
    dml_ref = <Model>(data, score=score)
    dml_ref.set_learners(...)
    dml_ref.draw_sample_splitting(n_folds=n_folds, n_rep=n_rep)
    dml_ref.fit()

    # 2. Build external_predictions dict from reference model
    external_predictions = {}
    if set_ml_x_ext:
        external_predictions["ml_x"] = dml_ref.predictions["ml_x"]

    # 3. Fit new model with shared splits and external predictions
    dml_ext = <Model>(data, score=score)
    dml_ext.set_learners(...)   # set non-external learners
    dml_ext._n_folds = n_folds
    dml_ext._n_rep = n_rep
    dml_ext._smpls = dml_ref.smpls
    dml_ext.fit(external_predictions=external_predictions)

    return {"ref": dml_ref, "ext": dml_ext}
```

### Required Tests

```python
import math

def test_coef(ext_pred_fixture):
    assert math.isclose(ref.coef[0], ext.coef[0], rel_tol=1e-9, abs_tol=1e-4)

def test_se(ext_pred_fixture):
    assert math.isclose(ref.se[0], ext.se[0], rel_tol=1e-9, abs_tol=1e-4)
```

Use `math.isclose` with `abs_tol=1e-4` instead of `np.testing.assert_allclose` because small numerical differences can accumulate when mixing external and fitted predictions.

---

## Assertion Patterns Summary

| Context | Assertion | Tolerance |
|---------|-----------|-----------|
| Comparison with old model | `np.testing.assert_allclose(new, old, rtol=1e-9)` | Exact match |
| External predictions | `math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-4)` | Small tolerance |
| Statistical accuracy | `abs(coef - true) <= 3.0 * se` | 3-sigma rule |
| Exception handling | `pytest.raises(Error, match=r"regex pattern")` | Exact message match |

---

## Fixture Scope Guidelines

| Scope | Use Case |
|-------|----------|
| `module` | Parametrized fixtures that fit models (expensive). Each parameter combination creates one instance shared across tests in the module. |
| `session` | Data generation that should be shared across all test modules (not typically needed for scalar model tests). |
| `function` | Only when test modifies state (rare for read-only assertion tests). |

---

## Checklist for New Scalar Models

When adding a new scalar model `<Model>` to the `DoubleMLScalar` hierarchy:

- [ ] **Implementation**: `doubleml/<module>/<model>_scalar.py`
  - [ ] Class inherits from `LinearScoreMixin` (or `NonLinearScoreMixin`)
  - [ ] `_LEARNER_SPECS` class variable defined
  - [ ] `required_learners` property returns score-dependent list
  - [ ] `set_learners()` with model-specific kwargs
  - [ ] `_check_data()` static method
  - [ ] `draw_sample_splitting()` (override if stratification needed)
  - [ ] `_nuisance_est()` per-fold estimation
  - [ ] `_get_score_elements()` returns `{psi_a, psi_b}`

- [ ] **Tests**: `doubleml/<module>/tests/`
  - [ ] `test_<model>_scalar.py` — core estimation
  - [ ] `test_<model>_scalar_return_types.py` — property shapes/types
  - [ ] `test_<model>_scalar_exceptions.py` — input validation
  - [ ] `test_<model>_scalar_vs_<model>.py` — comparison with old implementation
  - [ ] `test_<model>_scalar_external_predictions.py` — external predictions

- [ ] **Verification**
  - [ ] All new tests pass: `pytest doubleml/<module>/tests/test_<model>_scalar*.py -v -m ci`
  - [ ] Lint: `ruff check doubleml/<module>/<model>_scalar.py`
  - [ ] Format: `black doubleml/<module>/<model>_scalar.py`
  - [ ] Type check: `mypy doubleml/<module>/<model>_scalar.py`
  - [ ] Old tests still pass: `pytest doubleml/<module>/tests/ -v`

- [ ] **Documentation**: Update `doc/diagrams/architecture.md` class hierarchy
