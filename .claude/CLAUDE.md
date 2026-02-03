# DoubleML for Python - Claude Code Memory

## Project Purpose

DoubleML is a Python package implementing Double/Debiased Machine Learning (DML) methods for causal inference. The package provides:
- Partially Linear Models (PLR, PLIV, PLPR, LPLR)
- Interactive Regression Models (IRM, IIVM, APO, QTE, CVAR, SSM)
- Difference-in-Differences estimators (DID, DIDCSBinary, DIDMulti)
- Regression Discontinuity Design (RDD)

**Documentation**: https://docs.doubleml.org

## Coding Standards

### Python
- **Version**: Python 3.11+ (supports 3.11, 3.12, 3.13)
- **Formatter**: black with line-length 127
- **Linter**: ruff (rules: E, F, W, I)
- **Type Checker**: mypy with `disallow_untyped_defs = true`
- **Type hints**: Required for all functions
- **Docstrings**: NumPy-style (see example below)
- **Max line length**: 127 characters

### NumPy Docstring Style
```python
def example_function(param1: int, param2: str) -> bool:
    """
    Short description of the function.

    Parameters
    ----------
    param1 : int
        Description of param1.
    param2 : str
        Description of param2.

    Returns
    -------
    bool
        Description of return value.

    Raises
    ------
    ValueError
        If param1 is negative.
    """
```

### Code Quality Commands
```bash
# Format code
black .

# Lint code
ruff check .

# Fix linting issues
ruff check --fix .

# Type check
mypy doubleml
```

### Pre-commit Hooks
Pre-commit is configured with:
- File format checks (yaml, toml)
- Debug statement detection
- Large file checks
- Trailing whitespace and line ending fixes
- black formatting
- ruff linting with auto-fix

Run pre-commit manually: `pre-commit run --all-files`

## Architecture Overview

### Class Hierarchy
```
DoubleMLBase (ABC)
└─> DoubleMLScalar (ABC) - single-parameter models
    ├─> LinearScoreMixin - closed-form solver
    │   ├─> DoubleMLPLR
    │   ├─> DoubleMLIRM
    │   ├─> DoubleMLPLIV
    │   ├─> DoubleMLIIVM
    │   └─> DoubleML DID variants
    └─> NonLinearScoreMixin - numerical solver (planned)

DoubleML - multi-parameter estimation (extends DoubleMLScalar)
```

### Key Design Patterns
- **Template Method**: `fit()` orchestrates; subclasses implement abstract methods
- **Mixin Pattern**: LinearScoreMixin provides closed-form θ = -E[ψ_b]/E[ψ_a]
- **Delegation**: DoubleMLBase delegates inference to DoubleMLFramework

### Core Files
| File | Purpose |
|------|---------|
| `doubleml/double_ml_base.py` | Abstract base with properties (coef, se, summary) and inference methods |
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

## Testing

### Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov

# Run specific marker (CI tests)
pytest -m ci

# Run specific test file
pytest doubleml/tests/test_framework.py

# Run tests for a specific module
pytest doubleml/plm/tests/
```

### Test Markers
- `ci`: Continuous integration tests for GitHub Actions
- `ci_rdd`: RDD-specific CI tests

### Test Organization
- Each module (plm, irm, did) has its own `tests/` subdirectory
- Test utilities in `doubleml/tests/_utils*.py`
- Manual computation helpers verify results independently

## Git Workflow

### Branches
- `main`: Main development branch
- Feature branches for new work

### Commit Format
Use Conventional Commits:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation
- `refactor:` code refactoring
- `test:` adding tests
- `chore:` maintenance

## Key Dependencies

### Core
- numpy>=2.0.0, pandas>=2.0.0, scipy>=1.7.0
- scikit-learn>=1.6.0, statsmodels>=0.14.0

### ML/Tuning
- optuna>=4.6.0 (hyperparameter tuning)
- joblib>=1.2.0 (parallelization)

### Visualization
- matplotlib>=3.9.0, seaborn>=0.13, plotly>=5.0.0

### Development
- pytest>=8.3.0, pytest-cov>=6.0.0
- black>=25.1.0, ruff>=0.11.1, mypy>=1.18.0
- xgboost>=2.1.0, lightgbm>=4.6.0 (for testing)

## Known Pitfalls

### Type Annotations
- MyPy is strict: `disallow_untyped_defs = true`
- All functions need full type hints including return types
- Use `from __future__ import annotations` for forward references

### Learner Validation
- Learners must be scikit-learn compatible (fit/predict interface)
- Use `_check_learner()` from `doubleml/utils/_checks.py` for validation
- Classifiers need `predict_proba()` for propensity scores

### Sample Splitting
- Cross-fitting uses `DoubleMLResampling` from `doubleml/utils/resampling.py`
- Default is 5-fold cross-fitting with 1 repetition
- Cluster-robust resampling available for clustered data

### Score Functions
- Linear scores use closed-form: θ = -E[ψ_b]/E[ψ_a]
- Custom scores can be passed as callables
- Score elements: `psi_a` (derivative), `psi_b` (moment)

### External Predictions
- Models support external predictions via `set_external_predictions()`
- Predictions must match sample splitting structure

## Verification

Before completing any task:
1. Run `ruff check .` to check for linting issues
2. Run `mypy doubleml` for type checking
3. Run relevant tests: `pytest doubleml/path/to/tests/`
4. Format code: `black .`

## Useful Links

- **Documentation**: https://docs.doubleml.org
- **Source**: https://github.com/DoubleML/doubleml-for-py
- **Bug Tracker**: https://github.com/DoubleML/doubleml-for-py/issues
- **Architecture Docs**: [doc/diagrams/architecture.md](doc/diagrams/architecture.md)

---
*Update this file when Claude makes mistakes to prevent future issues.*
