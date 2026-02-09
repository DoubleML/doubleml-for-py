# Code style (summary)

- Module-level docstring required (one sentence).
- NumPy-style docstrings for public APIs.
- Full type hints, including return types.
- Use built-in generics (list[int], dict[str, T]) for Python 3.10+.
- Follow DoubleML patterns: `_check_*` helpers, `DoubleMLResampling`, clone learners.
- Score outputs use `psi_a` and `psi_b` with shape `(n_obs,)`.

Canonical: .claude/rules/py-code-conventions.md
