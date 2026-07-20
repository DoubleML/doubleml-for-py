# Copilot instructions for DoubleML (Python)

These instructions guide code and test authoring. Keep changes aligned with the detailed rules in .claude to avoid drift.

## Scope
- Authoring guidance only (not reviewer-only rules).
- Prefer concise, targeted edits; avoid unrelated refactors.

## Code style and design
- Start each Python file with a one-sentence module docstring.
- Use NumPy-style docstrings for public APIs (summary, Parameters, Returns).
- Require full type hints, including return types.
- Follow DoubleML patterns: `_check_*` validation helpers, `DoubleMLResampling` for splitting, clone learners before fit.
- Keep score outputs named `psi_a` and `psi_b` with shape `(n_obs,)`.

## Error handling
- Use `ValueError` for invalid values and `TypeError` for wrong types.
- Validate early (constructors and setters) with clear expected/actual messages.
- In tests, always use `pytest.raises(..., match=...)`.

## Testing
- Mark all tests with `@pytest.mark.ci`.
- Use module-scoped fixtures for expensive fits.
- Seed random generators for reproducibility.
- For new scalar models, follow the required 5-file test structure.

## Verification (lightweight)
Run relevant checks when changes warrant it:
- `black .`
- `ruff check --fix .`
- `mypy doubleml`
- `pytest -m ci`

## References (canonical rules)
- Code style: .claude/rules/py-code-conventions.md
- Error handling: .claude/rules/error-handling.md
- Performance: .claude/rules/performance-guidelines.md
- Testing: .claude/rules/testing-conventions.md
- Scalar test structure: .claude/rules/dml-scalar-test-structure.md

## Optional reference docs
- .github/copilot/code-style.md
- .github/copilot/error-handling.md
- .github/copilot/performance.md
- .github/copilot/testing.md
- .github/copilot/scalar-tests.md
