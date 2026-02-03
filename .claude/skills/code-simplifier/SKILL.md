---
name: code-simplifier
description: Simplify and clean up DoubleML code after changes. Reduces complexity, improves readability, ensures NumPy-style docstrings and type hints.
---

# Code Simplifier for DoubleML

Clean up and simplify code after making changes.

## When to Use

Run after completing a feature or fix to ensure code is clean, readable, and follows DoubleML patterns.

## Simplification Goals

### Reduce Complexity
- Break long functions into smaller, focused ones (target: <50 lines)
- Reduce nesting depth (max 3 levels)
- Simplify complex conditionals
- Extract magic numbers to named constants (e.g., `DEFAULT_N_FOLDS = 5`)

### Improve Readability
- Use descriptive variable and function names
- Add clarifying comments for non-obvious logic
- Ensure consistent formatting (127 char line limit)
- Remove unnecessary comments

### Apply Pythonic Patterns
- Use list/dict/set comprehensions where appropriate
- Use `with` statements for resource management
- Use `enumerate()` instead of manual indexing
- Use `zip()` for parallel iteration
- Use f-strings for formatting
- Use `pathlib` for file paths
- Use `is None` / `is not None` instead of `== None`

### DoubleML-Specific Patterns
- Use `clone()` for sklearn learners instead of direct copy
- Use `_check_learner()` for learner validation
- Use `_check_score()` for score function validation
- Consistent `psi_a`/`psi_b` naming for score elements
- Use `DoubleMLResampling` for sample splitting logic
- Prefer numpy operations over Python loops for arrays

### Type Hints (Python 3.11+)
- Use built-in generics: `list[int]` not `typing.List[int]`
- Use `X | None` instead of `Optional[X]`
- Use `X | Y` instead of `Union[X, Y]`
- Add `from __future__ import annotations` for forward references
- Ensure all public functions have complete type hints

### NumPy-Style Docstrings
- Ensure `Parameters` section lists all arguments
- Ensure `Returns` section describes return value
- Add `Raises` section for exceptions
- Use `:class:` references for DoubleML types

### Clean Up
- Remove unused imports
- Remove unused variables
- Remove commented-out code
- Remove redundant code paths
- Consolidate duplicate logic

## Workflow

1. **Identify Changed Files**
   ```bash
   git diff --name-only HEAD~1  # Recent changes
   git status --short           # Uncommitted changes
   ```

2. **Analyze Each File**
   - Check for simplification opportunities
   - Prioritize high-impact improvements

3. **Apply Simplifications**
   - Make incremental changes
   - Preserve original behavior
   - Run tests after each change

4. **Format and Lint**
   ```bash
   black .
   ruff check --fix .
   ```

5. **Type Check**
   ```bash
   mypy doubleml
   ```

6. **Verify**
   ```bash
   pytest -m ci
   ```

## Arguments

Optionally specify files or directories to simplify.

Usage:
- `/code-simplifier` - Simplify recently changed files
- `/code-simplifier doubleml/plm/plr.py` - Simplify specific file
- `/code-simplifier doubleml/utils/` - Simplify entire directory

## Example Transformations

### Loop to Comprehension
```python
# Before
result = []
for i in range(len(items)):
    if items[i].is_valid == True:
        result.append(items[i].value)

# After
result = [item.value for item in items if item.is_valid]
```

### Flatten Nesting
```python
# Before
if x != None:
    if y != None:
        if z != None:
            process(x, y, z)

# After
if all(v is not None for v in (x, y, z)):
    process(x, y, z)
```

### Modern Type Hints
```python
# Before
from typing import List, Optional, Union, Dict

def process(items: List[int], config: Optional[Dict[str, Any]] = None) -> Union[int, None]:
    ...

# After
def process(items: list[int], config: dict[str, Any] | None = None) -> int | None:
    ...
```

### NumPy Operations
```python
# Before
result = []
for i in range(len(predictions)):
    result.append(predictions[i] - true_values[i])
result = np.array(result)

# After
result = predictions - true_values
```

### DoubleML Learner Pattern
```python
# Before
ml_l_copy = copy.deepcopy(ml_l)

# After
from sklearn.base import clone
ml_l_copy = clone(ml_l)
```

### Score Element Naming
```python
# Before
def _get_score_elements(self, ...):
    return {"a": psi_derivative, "b": psi_moment}

# After
def _get_score_elements(self, ...):
    return {"psi_a": psi_derivative, "psi_b": psi_moment}
```
