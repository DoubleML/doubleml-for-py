# DoubleML for Python

DoubleML is a Python package implementing Double/Debiased Machine Learning (DML) methods for causal inference:
- Partially Linear Models (PLR, PLIV, PLPR, LPLR)
- Interactive Regression Models (IRM, IIVM, APO, QTE, CVAR, SSM)
- Difference-in-Differences estimators (DID, DIDCSBinary, DIDMulti)
- Regression Discontinuity Design (RDD)

**Docs**: https://docs.doubleml.org | **Source**: https://github.com/DoubleML/doubleml-for-py

## Architecture

### Class Hierarchy
```
DoubleMLBase (ABC)
└─> DoubleMLScalar (ABC) - single-parameter models
    ├─> LinearScoreMixin - closed-form solver (θ = -E[ψ_b]/E[ψ_a])
    │   ├─> DoubleMLPLR
    │   ├─> DoubleMLIRM
    │   ├─> DoubleMLPLIV
    │   ├─> DoubleMLIIVM
    │   └─> DoubleML DID variants
    └─> NonLinearScoreMixin - numerical solver (planned)

DoubleML - multi-parameter estimation (extends DoubleMLScalar)
```

### Design Patterns
- **Template Method**: `fit()` orchestrates; subclasses implement `_nuisance_est()`, `_get_score_elements()`
- **Mixin Pattern**: `LinearScoreMixin` provides closed-form coefficient estimation
- **Delegation**: `DoubleMLBase` delegates inference to `DoubleMLFramework`

### Core Files
| File | Purpose |
|------|---------|
| `doubleml/double_ml_base.py` | Abstract base with properties (coef, se, summary) and inference |
| `doubleml/double_ml_scalar.py` | Single-parameter estimation orchestrator |
| `doubleml/double_ml.py` | Multi-parameter estimation with sample splitting |
| `doubleml/double_ml_framework.py` | Statistical inference (confint, bootstrap, sensitivity) |
| `doubleml/double_ml_linear_score.py` | Linear score mixin |

### Package Structure
```
doubleml/
├── data/          # Data containers (DoubleMLData, DoubleMLDIDData, etc.)
├── plm/           # Partially Linear Models (PLR, PLIV, PLPR, LPLR)
├── irm/           # Interactive Regression Models (IRM, IIVM, APO, QTE, etc.)
├── did/           # Difference-in-Differences estimators
├── rdd/           # Regression Discontinuity Design
├── utils/         # Helpers (_checks, _estimation, resampling, tuning)
└── tests/         # Main test directory
```

## Key Dependencies

**Core**: numpy>=2.0.0, pandas>=2.0.0, scipy>=1.7.0, scikit-learn>=1.6.0, statsmodels>=0.14.0
**ML/Tuning**: optuna>=4.6.0, joblib>=1.2.0
**Visualization**: matplotlib>=3.9.0, seaborn>=0.13, plotly>=5.0.0
**Dev**: pytest>=8.3.0, black>=25.1.0, ruff>=0.11.1, mypy>=1.18.0, xgboost>=2.1.0, lightgbm>=4.6.0

## Git Workflow

- **Main branch**: `main`
- **Commits**: Conventional Commits — `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`

## Verification

Before completing any task:
```bash
black .                    # Format
ruff check --fix .         # Lint
mypy doubleml              # Type check
pytest -m ci               # Tests
```

## Coding Standards

Detailed conventions are in `.claude/rules/`:
- **py-code-conventions.md** — Formatting, type hints, docstrings, naming, DML-specific patterns
- **error-handling.md** — Exception types, validation patterns, warnings vs. errors
- **performance-guidelines.md** — Vectorization, pre-allocation, DML computation patterns
- **testing-conventions.md** — Markers, fixtures, assertion patterns
- **dml-scalar-test-structure.md** — Mandatory 5-file test structure for scalar models
