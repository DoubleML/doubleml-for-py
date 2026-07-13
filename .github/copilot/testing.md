# Testing (summary)

- Mark all tests with `@pytest.mark.ci`.
- Use module-scoped fixtures for expensive fits.
- Seed RNGs for reproducibility.
- Use `match=` in exception tests.
- Follow naming: `test_<model>_scalar*.py` for scalar models.
- Add conditional scalar test files for tuning, evaluate_learners, and sensitivity when those capabilities are implemented.

Canonical: .claude/rules/testing-conventions.md
