# Error handling (summary)

- Invalid values -> `ValueError`; wrong types -> `TypeError`.
- Validate early in constructors and setters.
- Error messages include expected vs actual values.
- Prefer `_check_*` helpers from doubleml/utils/_checks.py.
- Tests must use `pytest.raises(..., match=...)`.

Canonical: .claude/rules/error-handling.md
