---
name: techdebt
description: Find and fix technical debt in DoubleML codebase. Checks for code smells, type issues, style violations, and outdated patterns.
---

# Technical Debt Finder for DoubleML

Identify and fix technical debt aligned with project standards.

## Project-Specific Checks

### Type Annotations (MyPy Strict Mode)
- Missing type hints on functions (`disallow_untyped_defs = true`)
- Missing return type annotations
- Use of `Any` where specific types are possible
- Old-style typing (`typing.List` → `list`, `typing.Dict` → `dict`)
- Missing `from __future__ import annotations` for forward references

### Docstrings (NumPy Style)
- Missing docstrings on public functions/classes
- Incorrect docstring format (must be NumPy-style)
- Missing `Parameters`, `Returns`, or `Raises` sections
- Outdated parameter documentation

### Code Style (Black + Ruff)
- Lines exceeding 127 characters
- Import ordering issues (ruff rule I)
- Unused imports (ruff rule F401)
- Undefined names (ruff rule F821)
- Old-style string formatting (use f-strings)

### Scikit-learn Compatibility
- Learners missing `fit()`/`predict()` interface
- Classifiers missing `predict_proba()` for propensity scores
- Missing `clone()` compatibility

### DoubleML Patterns
- Inconsistent use of `_check_learner()` for validation
- Missing score function validation with `_check_score()`
- Incorrect sample splitting structure
- Missing `psi_a`/`psi_b` score elements

## Workflow

1. **Run Automated Checks**
   ```bash
   # Type checking
   mypy doubleml

   # Linting
   ruff check .

   # Format check (dry-run)
   black --check .
   ```

2. **Scan for Code Smells**
   - Functions longer than 50 lines
   - More than 5 parameters
   - Deep nesting (> 3 levels)
   - Duplicate code blocks
   - Magic numbers without constants

3. **Check for Dead Code**
   - Unused imports
   - Unused functions/classes
   - Commented-out code blocks
   - Unreachable code paths

4. **Report Findings**
   Format: `file_path:line_number - [severity] description`

5. **Fix Issues**
   - Auto-fix with `ruff check --fix .`
   - Auto-format with `black .`
   - Manual fixes for type hints and docstrings

6. **Verify**
   ```bash
   ruff check .
   mypy doubleml
   pytest -m ci  # Run CI tests
   ```

## Severity Levels

| Severity | Description | Examples |
|----------|-------------|----------|
| **high** | Breaks CI or type safety | Missing type hints, mypy errors |
| **medium** | Style violations | Line length, import order |
| **low** | Code smells | Long functions, magic numbers |

## Arguments

Specify scope to focus the scan:

- `/techdebt` - Scan entire `doubleml/` package
- `/techdebt doubleml/plm/` - Scan PLM module
- `/techdebt doubleml/utils/_checks.py` - Scan specific file

## Output Format

```markdown
## Technical Debt Report

### High Severity
- `doubleml/plm/plr.py:45` - Missing return type annotation
- `doubleml/utils/_checks.py:123` - Type hint uses `typing.List`

### Medium Severity
- `doubleml/did/did.py:89` - Line exceeds 127 characters
- `doubleml/irm/irm.py:12` - Unused import `warnings`

### Low Severity
- `doubleml/double_ml.py:234` - Function has 67 lines (>50)
- `doubleml/utils/resampling.py:45` - Magic number `5` should be constant

### Fixed
- ✓ Auto-fixed 3 import ordering issues
- ✓ Auto-formatted 2 files with black

### Remaining
- 2 high severity items need manual fixes
- Consider refactoring `_nuisance_est()` in next session
```
