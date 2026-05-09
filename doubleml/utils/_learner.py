"""
Learner specification and validation utilities for DoubleML.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
from sklearn.base import clone, is_classifier, is_regressor


@dataclass(frozen=True)
class LearnerSpec:
    """
    Immutable specification for a learner requirement.

    Parameters
    ----------
    name : str
        Name of the learner (e.g., "ml_l", "ml_m").
    allow_regressor : bool
        Whether regressors are allowed. Default is ``True``.
    allow_classifier : bool
        Whether classifiers are allowed. Default is ``True``.
    binary_data_check : {"outcome", "treatment"} or None
        If specified, warns when using regressor with binary data.
        "outcome" checks binary_outcome, "treatment" checks binary_treatment.
        Default is ``None``.
    """

    name: str
    allow_regressor: bool = True
    allow_classifier: bool = True
    binary_data_check: Optional[Literal["outcome", "treatment"]] = None

    def __post_init__(self) -> None:
        if not (self.allow_regressor or self.allow_classifier):
            raise ValueError(f"LearnerSpec '{self.name}': at least one of allow_regressor or allow_classifier must be True.")


@dataclass
class LearnerInfo:
    """
    Mutable info about a registered learner.

    Parameters
    ----------
    learner : object
        The learner object (already cloned).
    is_classifier : bool
        Whether the learner is a classifier.
    """

    learner: Any
    is_classifier: bool

    @property
    def predict_method(self) -> str:
        """Return the appropriate prediction method name."""
        return "predict_proba" if self.is_classifier else "predict"


def _check_learner_interface(learner: Any, err_prefix: str) -> None:
    """Raise TypeError if learner is a class or lacks fit/set_params/get_params."""
    if isinstance(learner, type):
        raise TypeError(err_prefix + "provide an instance of a learner instead of a class.")
    for method in ("fit", "set_params", "get_params"):
        if not hasattr(learner, method):
            raise TypeError(err_prefix + f"{str(learner)} has no method .{method}().")


def _determine_learner_type(learner: Any, spec: LearnerSpec, warn_prefix: str) -> bool:
    """Return True if learner should be treated as classifier; warn if type is ambiguous."""
    if spec.allow_regressor and spec.allow_classifier:
        if is_classifier(learner):
            return True
        if is_regressor(learner):
            return False
        warnings.warn(
            warn_prefix
            + f"{str(learner)} is (probably) neither a regressor nor a classifier. "
            + "Method predict is used for prediction."
        )
        return False
    if spec.allow_classifier:
        if not is_classifier(learner):
            warnings.warn(warn_prefix + f"{str(learner)} is (probably) no classifier.")
        return True
    if not is_regressor(learner):
        warnings.warn(warn_prefix + f"{str(learner)} is (probably) no regressor.")
    return False


def _check_binary_data_compatibility(
    learner: Any,
    spec: LearnerSpec,
    learner_is_classifier: bool,
    binary_outcome: bool,
    binary_treatment: bool,
) -> None:
    """Raise on classifier with non-binary data; warn on regressor with binary data."""
    if not spec.binary_data_check:
        return

    is_outcome_check = spec.binary_data_check == "outcome"
    data_is_binary = binary_outcome if is_outcome_check else binary_treatment
    var_label = "outcome" if is_outcome_check else "treatment"

    if learner_is_classifier and not data_is_binary:
        raise ValueError(
            f"The {spec.name} learner {str(learner)} was identified as classifier "
            f"but the {var_label} variable is not binary with values 0 and 1."
        )

    if not learner_is_classifier and data_is_binary:
        action = "fit an additive probability model" if is_outcome_check else "estimate propensity scores"
        warnings.warn(
            f"Binary {var_label} detected. Consider using a classifier for {spec.name} " f"with predict_proba() to {action}."
        )


def validate_learner(
    learner: Any,
    spec: LearnerSpec,
    binary_outcome: bool = False,
    binary_treatment: bool = False,
) -> LearnerInfo:
    """
    Validate learner against specification and data properties.

    Parameters
    ----------
    learner : object
        The learner to validate.
    spec : LearnerSpec
        Specification for this learner.
    binary_outcome : bool
        Whether the outcome variable is binary.
    binary_treatment : bool
        Whether the treatment variable is binary.

    Returns
    -------
    LearnerInfo
        Information about the validated learner.

    Raises
    ------
    TypeError
        If the learner is a class instead of an instance, or lacks
        required methods (fit, set_params, get_params, predict/predict_proba).
    ValueError
        If the learner type is not allowed by the specification.
        If a classifier is used with non-binary data when required.
    """
    err_msg_prefix = f"Invalid learner provided for {spec.name}: "
    warn_msg_prefix = f"Learner provided for {spec.name} is probably invalid: "

    _check_learner_interface(learner, err_msg_prefix)
    learner_is_classifier = _determine_learner_type(learner, spec, warn_msg_prefix)

    # Check type is allowed by spec
    if learner_is_classifier and not spec.allow_classifier:
        raise ValueError(f"Classifier not allowed for {spec.name}. Use a regressor instead.")
    if not learner_is_classifier and not spec.allow_regressor:
        raise ValueError(f"Regressor not allowed for {spec.name}. Use a classifier instead.")

    # Check prediction method exists
    predict_method = "predict_proba" if learner_is_classifier else "predict"
    if not hasattr(learner, predict_method):
        raise TypeError(err_msg_prefix + f"{str(learner)} has no method .{predict_method}().")

    _check_binary_data_compatibility(learner, spec, learner_is_classifier, binary_outcome, binary_treatment)

    return LearnerInfo(
        learner=clone(learner),
        is_classifier=learner_is_classifier,
    )


def predict_nuisance(learner: Any, X: np.ndarray, is_classifier: bool) -> np.ndarray:
    """
    Predict using the appropriate method based on learner type.

    Parameters
    ----------
    learner : object
        Fitted learner with predict() or predict_proba() method.
    X : np.ndarray
        Features to predict on.
    is_classifier : bool
        Whether the learner is a classifier.

    Returns
    -------
    np.ndarray
        Predictions. For classifiers, returns probability of class 1.
    """
    if is_classifier:
        return learner.predict_proba(X)[:, 1]
    return learner.predict(X)
