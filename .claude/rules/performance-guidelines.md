# Performance Guidelines — DoubleML

> **Apply when**: Writing nuisance estimation, score computation, or any code operating on `(n_obs,)` or `(n_obs, n_rep)` arrays.

## Core Rules

1. **Vectorize** — Use NumPy array operations, never Python loops over observations
2. **Pre-allocate** — Create output arrays at full size before filling per-fold
3. **Clone before fit** — `clone(learner).fit(X, y)` — learners are mutable
4. **Profile first** — Don't optimize without measuring

## DoubleML-Specific Patterns

### Nuisance Estimation (Per-Fold)

```python
# Pre-allocate prediction arrays
predictions = {
    'ml_l': np.zeros((n_obs, n_rep)),
    'ml_m': np.zeros((n_obs, n_rep)),
}

for i_rep, smpl in enumerate(smpls):
    for train_idx, test_idx in smpl:
        # Clone learner (mutable!), fit, predict in one chain
        predictions['ml_l'][test_idx, i_rep] = (
            clone(self._learner_ml_l).fit(X[train_idx], y[train_idx]).predict(X[test_idx])
        )
```

### Score Computation

```python
# Vectorized — operates on full arrays
psi_a = -d_res * d_res          # (n_obs,)
psi_b = d_res * (y - ml_g_hat)  # (n_obs,)
theta = -np.mean(psi_b) / np.mean(psi_a)
```

### Propensity Scores

```python
# predict_proba returns (n_obs, 2) — take column 1
propensity = clone(self._learner_ml_m).fit(X_train, d_train).predict_proba(X_test)[:, 1]

# Clip in one vectorized operation
propensity = np.clip(propensity, self._trimming_threshold, 1 - self._trimming_threshold)
```

### Matrix Operations

```python
# Use lstsq, not manual inversion
beta = np.linalg.lstsq(X, y, rcond=None)[0]

# Not: beta = np.linalg.inv(X.T @ X) @ X.T @ y  (numerically unstable)
```

## Anti-Patterns

| Don't | Do Instead |
|-------|-----------|
| `for i in range(n_obs): result[i] = ...` | `result = vectorized_op(array)` |
| `np.append(result, value)` in a loop | Pre-allocate `np.zeros(n)`, fill by index |
| `df.apply(lambda x: ...)` | `df['col'] ** 2` or `np.log(df['col'])` |
| `KFold(n_splits=5)` | `DoubleMLResampling(n_folds=5, ...)` |
| `np.linalg.inv(X.T @ X) @ X.T @ y` | `np.linalg.lstsq(X, y, rcond=None)[0]` |
