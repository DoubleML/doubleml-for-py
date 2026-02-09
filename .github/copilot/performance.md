# Performance (summary)

- Vectorize array operations; avoid Python loops over observations.
- Pre-allocate `(n_obs, n_rep)` arrays before filling.
- Clone learners before fit (mutable estimators).
- Use `DoubleMLResampling`, not raw `KFold`.
- Prefer `np.linalg.lstsq` over manual inversion.

Canonical: .claude/rules/performance-guidelines.md
