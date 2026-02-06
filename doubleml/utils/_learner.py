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

    # Check it's an instance, not a class
    if isinstance(learner, type):
        raise TypeError(err_msg_prefix + "provide an instance of a learner instead of a class.")

    # Check required methods
    if not hasattr(learner, "fit"):
        raise TypeError(err_msg_prefix + f"{str(learner)} has no method .fit().")
    if not hasattr(learner, "set_params"):
        raise TypeError(err_msg_prefix + f"{str(learner)} has no method .set_params().")
    if not hasattr(learner, "get_params"):
        raise TypeError(err_msg_prefix + f"{str(learner)} has no method .get_params().")

    # Determine learner type
    learner_is_classifier: bool
    if spec.allow_regressor and spec.allow_classifier:
        if is_classifier(learner):
            learner_is_classifier = True
        elif is_regressor(learner):
            learner_is_classifier = False
        else:
            warnings.warn(
                warn_msg_prefix
                + f"{str(learner)} is (probably) neither a regressor nor a classifier. "
                + "Method predict is used for prediction."
            )
            learner_is_classifier = False
    elif spec.allow_classifier:
        if not is_classifier(learner):
            warnings.warn(warn_msg_prefix + f"{str(learner)} is (probably) no classifier.")
        learner_is_classifier = True
    else:
        assert spec.allow_regressor  # At least one must be True
        if not is_regressor(learner):
            warnings.warn(warn_msg_prefix + f"{str(learner)} is (probably) no regressor.")
        learner_is_classifier = False

    # Check type is allowed
    if learner_is_classifier and not spec.allow_classifier:
        raise ValueError(f"Classifier not allowed for {spec.name}. Use a regressor instead.")
    if not learner_is_classifier and not spec.allow_regressor:
        raise ValueError(f"Regressor not allowed for {spec.name}. Use a classifier instead.")

    # Check prediction method exists
    if learner_is_classifier:
        if not hasattr(learner, "predict_proba"):
            raise TypeError(err_msg_prefix + f"{str(learner)} has no method .predict_proba().")
    else:
        if not hasattr(learner, "predict"):
            raise TypeError(err_msg_prefix + f"{str(learner)} has no method .predict().")

    # Check binary data compatibility for classifiers
    if learner_is_classifier and spec.binary_data_check:
        if spec.binary_data_check == "outcome" and not binary_outcome:
            raise ValueError(
                f"The {spec.name} learner {str(learner)} was identified as classifier "
                "but the outcome variable is not binary with values 0 and 1."
            )
        if spec.binary_data_check == "treatment" and not binary_treatment:
            raise ValueError(
                f"The {spec.name} learner {str(learner)} was identified as classifier "
                "but the treatment variable is not binary with values 0 and 1."
            )

    # Warn if regressor used with binary data
    if not learner_is_classifier and spec.binary_data_check:
        if spec.binary_data_check == "outcome" and binary_outcome:
            warnings.warn(
                f"Binary outcome detected. Consider using a classifier for {spec.name} "
                "with predict_proba() to fit an additive probability model."
            )
        elif spec.binary_data_check == "treatment" and binary_treatment:
            warnings.warn(
                f"Binary treatment detected. Consider using a classifier for {spec.name} "
                "with predict_proba() to estimate propensity scores."
            )

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
