# Scalar model test structure (summary)

New DoubleMLScalar models require five base test files:
- test_<model>_scalar.py
- test_<model>_scalar_return_types.py
- test_<model>_scalar_exceptions.py
- test_<model>_scalar_vs_<model>.py
- test_<model>_scalar_external_predictions.py

Additional conditional test files are required when the model implements:
- test_<model>_scalar_tune_ml_models.py - if the model has tune_ml_models()
- test_<model>_scalar_evaluate_learners.py - if the model has evaluate_learners() or nuisance_loss
- test_<model>_scalar_sensitivity.py - if the model supports sensitivity analysis

See details and required assertions in the canonical rule.

Canonical: .claude/rules/dml-scalar-test-structure.md
