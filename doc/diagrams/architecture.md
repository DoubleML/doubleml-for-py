# DoubleML Scalar Architecture

## Class Hierarchy

```
DoubleMLBase (ABC)
│   Data storage, framework delegation (coef, se, summary, confint, bootstrap, ...)
│
└── DoubleMLScalar (ABC)
    │   Single-parameter estimation: fit(), draw_sample_splitting(),
    │   fit_nuisance_models(), estimate_causal_parameters()
    │   Learner management: set_learners(), _check_learners_available()
    │   Prediction storage: _initialize_predictions_dict()
    │
    ├── LinearScoreMixin
    │   │   Implements _est_causal_pars_and_se() for linear scores
    │   │   θ̂ = -E[ψ_b] / E[ψ_a]
    │   │
    │   ├── PLR          (partialling out, IV-type)
    │   ├── IRM          (ATE, ATTE)
    │   ├── PLIV         (planned)
    │   └── DID          (planned)
    │
    └── NonLinearScoreMixin (planned)
        │   Implements _est_causal_pars_and_se() via numerical root-finding
        │
        └── ...
```

## UML Class Diagram

```
┌─────────────────────────────────────────┐
│          DoubleMLBase (ABC)             │
├─────────────────────────────────────────┤
│ - _dml_data: DoubleMLBaseData           │
│ - _n_obs: int                           │
│ - _framework: DoubleMLFramework | None  │
├─────────────────────────────────────────┤
│ + framework: DoubleMLFramework          │
│ + thetas / coef: np.ndarray             │
│ + all_thetas / all_coef: np.ndarray     │
│ + se: np.ndarray                        │
│ + all_ses: np.ndarray                   │
│ + summary: pd.DataFrame                 │
│ + psi: np.ndarray                       │
│ + n_obs: int                            │
│ + confint()                             │
│ + bootstrap()                           │
│ + p_adjust()                            │
│ + sensitivity_analysis()                │
│ «abstract» + fit()                      │
│ «abstract» + n_rep: int                 │
└─────────────────┬───────────────────────┘
                  │ inherits
┌─────────────────▼───────────────────────┐
│         DoubleMLScalar (ABC)            │
├─────────────────────────────────────────┤
│ - _score: str                           │
│ - _learner_names: List[str]             │
│ - _learners: Dict[str, object]          │
│ - _n_folds: int | None                  │
│ - _n_rep: int | None                    │
│ - _smpls: List | None                   │
│ - _predictions: Dict | None             │
│ - _all_thetas: np.ndarray | None        │
│ - _all_ses: np.ndarray | None           │
│ - _psi: np.ndarray | None               │
│ - _psi_deriv: np.ndarray | None         │
│ - _var_scaling_factors: np.ndarray|None │
├─────────────────────────────────────────┤
│ + score: str                            │
│ + n_folds: int                          │
│ + n_rep: int                            │
│ + predictions: Dict                     │
│ + smpls: List                           │
│ + learner_names: List[str]              │
│ + learners: Dict[str, object]           │
│ + fit(n_folds, n_rep, external_preds)   │
│ + fit_nuisance_models(external_preds)   │
│ + estimate_causal_parameters()          │
│ + draw_sample_splitting(n_folds, n_rep) │
│ + _initialize_predictions_dict()        │
│ + _check_learners_available()           │
│ + _initialize_result_arrays()           │
│ + _construct_framework()                │
│ «abstract» + set_learners()             │
│ «abstract» + _nuisance_est()            │
│ «abstract» + _get_score_elements()      │
│ «abstract» + _est_causal_pars_and_se()  │
└──────────┬──────────────────────────────┘
           │ inherits
┌──────────▼──────────────────────────────┐
│       LinearScoreMixin                  │
├─────────────────────────────────────────┤
│ (no additional state)                   │
├─────────────────────────────────────────┤
│ + _est_causal_pars_and_se(psi_elements) │
│   → closed-form: θ̂ = -E[ψ_b]/E[ψ_a]  │
│ + _compute_score(psi_elements, coef)    │
│ + _score_element_names() → [psi_a,b]   │
└──────────┬──────────────────────────────┘
           │ inherits
┌──────────▼──────────────────────────────┐
│              PLR                        │
├─────────────────────────────────────────┤
│ _learner_names = [ml_l, ml_m(, ml_g)]  │
├─────────────────────────────────────────┤
│ + __init__(obj_dml_data, score)         │
│ + set_learners(ml_l, ml_m, ml_g)       │
│ + _check_data()                         │
│ + _nuisance_est(train, test, i_rep, ..) │
│ + _get_score_elements() → {psi_a,psi_b}│
└─────────────────────────────────────────┘
```

## Method Resolution & Workflow

The `fit()` call follows the template method pattern:

```
PLR.fit()
  │
  ├─ DoubleMLScalar.draw_sample_splitting()    ← if not already done
  │    └─ DoubleMLResampling.split_samples()
  │
  ├─ DoubleMLScalar.fit_nuisance_models()
  │    ├─ DoubleMLScalar._check_learners_available()
  │    ├─ DoubleMLScalar._initialize_predictions_dict()  ← uses _learner_names
  │    └─ loop(n_rep × n_folds):
  │         └─ PLR._nuisance_est()                       ← subclass implements
  │
  └─ DoubleMLScalar.estimate_causal_parameters()
       ├─ DoubleMLScalar._initialize_result_arrays()
       ├─ PLR._get_score_elements()                      ← subclass implements
       ├─ LinearScoreMixin._est_causal_pars_and_se()     ← mixin implements
       └─ DoubleMLScalar._construct_framework()
            └─ DoubleMLFramework(...)
```

## Typical User Workflow

```python
# 1. Define model (data + score)
plr = PLR(obj_dml_data, score="partialling out")

# 2. Set learners
plr.set_learners(ml_l=RandomForestRegressor(), ml_m=RandomForestRegressor())

# 3. Draw sample splitting
plr.draw_sample_splitting(n_folds=5, n_rep=1)

# 4. Fit
plr.fit()

# 5. Results (delegated to DoubleMLFramework via DoubleMLBase)
print(plr.summary)
plr.confint()
plr.bootstrap()
```

## What Each Layer Provides

| Layer | Responsibilities |
|---|---|
| **DoubleMLBase** | Data storage, framework delegation (coef, se, summary, confint, bootstrap, p_adjust, sensitivity_analysis) |
| **DoubleMLScalar** | Single-parameter fit orchestration, sample splitting, learner management (`_learner_names`, `_learners`, `set_learners`, `_check_learners_available`), prediction storage, result array initialization, framework construction |
| **LinearScoreMixin** | Closed-form parameter estimation for linear scores: `θ̂ = -E[ψ_b]/E[ψ_a]`, SE computation, influence function |
| **PLR** | PLR-specific: data validation, learner names (`ml_l`, `ml_m`, `ml_g`), nuisance estimation logic, score element computation |

## Key Design Decisions

- **Learners separated from constructor**: `__init__` takes only data + score; learners are set via `set_learners()` with explicit kwargs per subclass
- **`_learner_names` as single source of truth**: Drives `_initialize_predictions_dict()` and `_check_learners_available()` — subclasses just set the list
- **Resampling separated from constructor**: `draw_sample_splitting()` is a separate step, can be called independently
- **External predictions**: Passed to `fit()` / `fit_nuisance_models()`, validated against `_learner_names`, pre-filled before cross-fitting loop
- **Template method pattern**: `fit()` orchestrates; subclasses implement `_nuisance_est()` and `_get_score_elements()`; mixin implements `_est_causal_pars_and_se()`
