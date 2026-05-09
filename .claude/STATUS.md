# Branch Status & TODOs

> Tracked in git so it syncs across machines. Update this file as work progresses.
> Reference: `CLAUDE.md` loads this automatically via the line below.

---

## Branch: `sk-refactoring`

**Goal**: Introduce a new `DoubleMLScalar` / `DoubleMLVector` hierarchy alongside
the existing `DoubleML` API — cleaner design, better testability, explicit tuning,
nuisance evaluation, and sensitivity analysis.

### Completed

- [x] **Claude tooling** — `.claude/` dir, `CLAUDE.md`, `rules/`, `agents/`, `skills/`
- [x] **Architecture docs** — `doc/diagrams/architecture.md`, `doc/diagrams/testing_structure.md`
- [x] **`DoubleMLBase`** — abstract base with shared properties (`coef`, `se`, `summary`) and inference delegation (`doubleml/double_ml_base.py`)
- [x] **`LinearScoreMixin`** — closed-form θ = −E[ψ_b]/E[ψ_a] solver (`doubleml/double_ml_linear_score.py`)
- [x] **`DoubleMLScalar`** — single-parameter orchestrator (`doubleml/double_ml_scalar.py`) with:
  - `fit()` → `draw_sample_splitting()` + `fit_nuisance_models()` + `estimate_causal_parameters()`
  - `tune_ml_models()` via Optuna (`_LEARNER_PARAM_ALIASES`, `_get_tuning_data()` hook)
  - `nuisance_targets`, `nuisance_loss`, `evaluate_learners()`
  - `_sensitivity_element_est()` hook + full sensitivity analysis pipeline
- [x] **`DoubleMLPLRScalar`** — PLR scalar (`doubleml/plm/plr_scalar.py`) with all 7 test files:
  - `test_plr_scalar.py`, `_return_types`, `_exceptions`, `_vs_plr`, `_external_predictions`, `_tune_ml_models`, `_evaluate_learners`, `_sensitivity`
- [x] **`DoubleMLIRMScalar`** — IRM scalar (`doubleml/irm/irm_scalar.py`) with all 7 test files (same structure)
- [x] **`cate()` + `gate()` for IRM scalar** — `doubleml/irm/irm_scalar.py` + `test_irm_scalar_cate_gate.py`
- [x] **`cate()` + `gate()` + `_partial_out()` for PLR scalar** — `doubleml/plm/plr_scalar.py` + `test_plr_scalar_cate_gate.py`. Multi-rep × multi-column basis fully supported.
- [x] **`DoubleMLBLP` per-rep basis API** — `basis` may be a single `pd.DataFrame` (shared) or a `list[pd.DataFrame]` of length `n_rep`. Also fixes the legacy `DoubleMLPLR.cate()` multi-rep bug (`basis * D_tilde` mis-broadcast for `n_rep>1` and `d_basis>1`).
- [x] **`DoubleMLVector`** — multi-treatment base class first iteration (`doubleml/double_ml_vector.py`)
- [x] **BLP multi-rep support** — `doubleml/utils/blp.py`

### In Progress

- [ ] **`DoubleMLVector`** — base class exists; no concrete subclass yet

### Feature Gaps vs Legacy Classes

Missing from `PLR` / `IRM` scalar compared to `DoubleMLPLR` / `DoubleMLIRM`:

| Feature | Legacy location | Applies to | Notes |
|---------|----------------|-----------|-------|
| `cate()` | `plr.py:447`, `irm.py:564` | — | ✅ ported for both IRM and PLR |
| `gate()` | `plr.py:485`, `irm.py:598` | — | ✅ ported for both IRM and PLR |
| `_partial_out()` | `plr.py:522` | — | ✅ ported for PLR scalar |
| `policy_tree()` | `irm.py:635` | IRM only | Not planned yet |

Weighted effects in IRM (`weights` dict form):
- Array weights: ✅ supported
- Dict weights with `weights_bar`: ✅ supported — init defers the `n_rep` column check; `DoubleMLScalar._check_smpls_dependent_inputs()` hook validates `weights_bar.shape == (n_obs, n_rep)` from inside both `draw_sample_splitting()` and `set_sample_splitting()`. `fit(n_folds=..., n_rep=...)` re-draws splits with a `UserWarning` when args conflict with existing splits.

Intentionally **not ported**:
- Callable score — design decision
- `trimming_rule` / `trimming_threshold` deprecated props — use `ps_processor_config`

### Planned

| Item | Files | Notes |
|------|-------|-------|
| `DoubleMLPLRVector` | `doubleml/plm/plr_vector.py` + tests | First concrete Vector subclass |
| `DoubleMLPLIVScalar` | `doubleml/plm/pliv_scalar.py` + 7 test files | Next scalar model |
| `DoubleMLPLPRScalar` | `doubleml/plm/plpr_scalar.py` + 7 test files | |
| DID scalar variants | `doubleml/did/*_scalar.py` | DID, DIDCSBinary, DIDMulti |
| `DoubleMLVector` tests | `doubleml/tests/test_vector_*.py` | Base class tests |

---

## How to Update This File

- Mark items `[x]` when complete
- Move items between sections as work progresses
- Add new planned items as they are identified
- Commit this file with the relevant code changes so the status stays in sync
