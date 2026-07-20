# DoubleMLScalar Test Structure

> **Apply when**: Implementing a new model in the `DoubleMLScalar` hierarchy.
> **Source**: Derived from `doc/diagrams/testing_structure.md`.

## Required Test Files

Every scalar model `<model>` in module `<module>/` requires **5 test files** in `doubleml/<module>/tests/`:

| File                                          | Purpose                                 |
| --------------------------------------------- | --------------------------------------- |
| `test_<model>_scalar.py`                      | Core estimation accuracy (3-sigma rule) |
| `test_<model>_scalar_return_types.py`         | Property types, shapes, API contracts   |
| `test_<model>_scalar_exceptions.py`           | Input validation, error messages        |
| `test_<model>_scalar_vs_<model>.py`           | Exact match with old `DoubleML<Model>`  |
| `test_<model>_scalar_external_predictions.py` | External predictions equivalence        |

All test functions must be marked `@pytest.mark.ci`.

---

## 1. Core Estimation (`test_<model>_scalar.py`)

**Fixture**: Parametrize over `score` variants and model-specific options. Use `scope="module"`, `np.random.seed(3141)`, `n_obs=500`, `n_folds=5`, `n_rep=1`.

**Required tests**:
- `test_coef`: `abs(coef - true_theta) <= 3.0 * se` (when true theta matches DGP)
  - For unknown true params (e.g., ATTE): `np.isfinite(coef)` and `abs(coef) < 10.0`
- `test_se`: `se > 0`

## 2. Return Types (`test_<model>_scalar_return_types.py`)

**Constants**: `N_OBS=200`, `N_FOLDS=3`, `N_REP=2`. Single fixture fitting one model.

**Required tests**:

| Test                                     | Assertion                                                                     |
| ---------------------------------------- | ----------------------------------------------------------------------------- |
| `test_coef_type_and_shape`               | `isinstance(coef, np.ndarray)`, `shape == (1,)`                               |
| `test_se_type_and_shape`                 | `isinstance(se, np.ndarray)`, `shape == (1,)`                                 |
| `test_all_thetas_shape`                  | `shape == (1, N_REP)`                                                         |
| `test_all_ses_shape`                     | `shape == (1, N_REP)`                                                         |
| `test_summary_type`                      | `isinstance(summary, pd.DataFrame)`, `len == 1`                               |
| `test_confint_type_and_shape`            | `isinstance(ci, pd.DataFrame)`, `shape == (1, 2)`                             |
| `test_psi_shape`                         | `shape == (N_OBS, 1, N_REP)`                                                  |
| `test_predictions_type`                  | `isinstance(predictions, dict)`, each value `shape == (N_OBS, N_REP)`         |
| `test_smpls_type`                        | `len(smpls) == N_REP`, each has `N_FOLDS` tuples of `(train, test)` arrays    |
| `test_n_properties`                      | `n_obs == N_OBS`, `n_folds == N_FOLDS`, `n_rep == N_REP`, `score == expected` |
| `test_required_learners`                 | Returns list of expected learner names                                        |
| `test_str_repr`                          | `str(model)` and `repr(model)` return `str`                                   |
| `test_get_params`                        | Returns dict with learner keys                                                |
| `test_set_params`                        | Modifies and confirms learner parameter change                                |
| `test_before_fit_raises`                 | `coef`/`se` before `fit()` raises error                                       |
| `test_reset_after_set_learners`          | Updating learners clears fitted results                                       |
| `test_reset_after_draw_sample_splitting` | Changing splits clears fitted results                                         |

## 3. Exceptions (`test_<model>_scalar_exceptions.py`)

**Common exception tests** (required for all models):

| Test                                                 | Input                                   | Expected     |
| ---------------------------------------------------- | --------------------------------------- | ------------ |
| `test_exception_data`                                | Non-DoubleMLData                        | `TypeError`  |
| `test_exception_score`                               | Invalid score string                    | `ValueError` |
| `test_exception_n_folds`                             | `n_folds < 2`                           | `ValueError` |
| `test_exception_n_rep`                               | `n_rep < 1`                             | `ValueError` |
| `test_exception_fit_nuisance_without_smpls`          | Fit before `draw_sample_splitting()`    | `ValueError` |
| `test_exception_estimate_causal_without_predictions` | Estimate before `fit_nuisance_models()` | `ValueError` |
| `test_exception_missing_learner`                     | `fit()` without required learners       | `ValueError` |
| `test_exception_invalid_learner`                     | Class instead of instance               | `TypeError`  |

**Model-specific exceptions** to add per model:
- PLR: multiple treatments, `ml_g` warning for partialling out
- IRM: non-binary treatment, `ml_m` must be classifier, `normalize_ipw` type

Always use `pytest.raises(Error, match=r"regex pattern")`.

## 4. Comparison (`test_<model>_scalar_vs_<model>.py`)

**Fixture**: Parametrize `score` and `n_rep` (use `[1, 3]`).

**Critical pattern**: Share sample splits from old model:
```python
dml_new._smpls = dml_old.smpls  # Old/new consume random state differently
```

**Required tests** — all use `np.testing.assert_allclose(..., rtol=1e-9)`:
- `test_coef_equal`: `new.coef` vs `old.coef`
- `test_se_equal`: `new.se` vs `old.se`
- `test_all_coef_equal`: `new.all_thetas` vs `old.all_coef` (note: property name differs!)
- `test_all_se_equal`: `new.all_ses` vs `old.all_se`

## 5. External Predictions (`test_<model>_scalar_external_predictions.py`)

**Fixture**: Parametrize `score`, `n_rep` (`[1, 3]`), and one `set_ml_x_ext` bool fixture per learner.

**Pattern**:
1. Fit reference model normally
2. Extract `dml_ref.predictions['ml_x']` for external learners
3. Fit test model with `dml_ext._smpls = dml_ref.smpls` and `fit(external_predictions=...)`

**Required tests** — use `math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-4)` (not `assert_allclose`):
- `test_coef`: Reference vs. external
- `test_se`: Reference vs. external

`math.isclose` with `abs_tol=1e-4` because small numerical differences accumulate when mixing external and fitted predictions.

---

## Assertion Tolerance Summary

| Context                | Method                                     | Why                    |
| ---------------------- | ------------------------------------------ | ---------------------- |
| Core estimation        | `abs(coef - true) <= 3.0 * se`             | Statistical 3-sigma    |
| Backward compatibility | `assert_allclose(rtol=1e-9)`               | Must be identical      |
| External predictions   | `math.isclose(rel_tol=1e-9, abs_tol=1e-4)` | Numerical accumulation |

## New Model Checklist

### Implementation
- [ ] Inherits from `LinearScoreMixin` (or `NonLinearScoreMixin`)
- [ ] `_LEARNER_SPECS` class variable defined
- [ ] `required_learners` property (score-dependent list)
- [ ] `set_learners()` with model-specific kwargs
- [ ] `_check_data()` static method
- [ ] `draw_sample_splitting()` (override if stratification needed)
- [ ] `_nuisance_est()` per-fold estimation
- [ ] `_get_score_elements()` returns `{'psi_a': ..., 'psi_b': ...}`

### Tests
- [ ] All 5 test files created and pass: `pytest doubleml/<module>/tests/test_<model>_scalar*.py -v -m ci`
- [ ] Old tests still pass: `pytest doubleml/<module>/tests/ -v`

### Quality
- [ ] `black doubleml/<module>/`
- [ ] `ruff check doubleml/<module>/`
- [ ] `mypy doubleml/<module>/`
