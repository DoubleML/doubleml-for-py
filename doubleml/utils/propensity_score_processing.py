import warnings
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict
from sklearn.utils.multiclass import type_of_target


@dataclass
class PSProcessorConfig:
    """
    Configuration for propensity score processing.

    This dataclass holds the configuration parameters used by PSProcessor
    for propensity score calibration, clipping, and validation.

    Parameters
    ----------
    clipping_threshold : float, default=1e-2
        Minimum and maximum bound for propensity scores after clipping.
        Must be between 0 and 0.5.

    extreme_threshold : float, default=1e-12
        Threshold below which propensity scores are considered extreme.
        Propensity scores are clipped based on this value when scores are too close to 0 or 1
        to avoid numerical instability.
        Must be between 0 and 0.5.

    calibration_method : {'isotonic', None}, optional
        If provided, applies the specified calibration method to
        the propensity scores before clipping. Currently supports:
        - 'isotonic': Isotonic regression calibration
        - None: No calibration applied

    cv_calibration : bool, default=False
        Whether to use cross-validation for calibration.
        Only applies if a calibration method is specified.
        Requires calibration_method to be set.

    Examples
    --------
    >>> from doubleml.utils import PSProcessorConfig, PSProcessor
    >>> config = PSProcessorConfig(
    ...     clipping_threshold=0.05,
    ...     calibration_method='isotonic',
    ...     cv_calibration=True
    ... )
    >>> processor = PSProcessor.from_config(config)
    """

    clipping_threshold: float = 1e-2
    extreme_threshold: float = 1e-12
    calibration_method: Optional[str] = None
    cv_calibration: bool = False


# TODO [v0.12.0]: Remove support for 'trimming_rule' and 'trimming_threshold' (deprecated).
def init_ps_processor(
    ps_processor_config: Optional[PSProcessorConfig], trimming_rule: Optional[str], trimming_threshold: Optional[float]
):
    if trimming_rule != "truncate":
        warnings.warn(
            "'trimming_rule' is deprecated and will be removed in a future version. "
            "Use 'ps_processor_config' with 'clipping_threshold' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    if trimming_threshold != 1e-2:
        warnings.warn(
            "'trimming_threshold' is deprecated and will be removed in a future version. "
            "Use 'ps_processor_config' with 'clipping_threshold' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    if ps_processor_config is not None:
        config = ps_processor_config
    else:
        config = PSProcessorConfig(clipping_threshold=trimming_threshold if trimming_threshold is not None else 1e-2)
    processor = PSProcessor.from_config(config)
    return config, processor


class PSProcessor:
    """
    Processor for propensity score calibration, clipping, and validation.

    Parameters
    ----------
    clipping_threshold : float, default=1e-2
        Minimum and maximum bound for propensity scores after clipping.

    extreme_threshold : float, default=1e-12
        Threshold below which propensity scores are considered extreme.
        Used for generating warnings.

    calibration_method : {'isotonic', None}, optional
        If provided, applies the specified calibration method to
        the propensity scores before clipping.

    cv_calibration : bool, default=False
        Whether to use cross-validation for calibration.
        Only applies if a calibration method is specified.

    Examples
    --------
    >>> import numpy as np
    >>> from doubleml.utils import PSProcessor
    >>> ps = np.array([0.001, 0.2, 0.5, 0.8, 0.999])
    >>> treatment = np.array([0, 1, 1, 0, 1])
    >>> processor = PSProcessor(clipping_threshold=0.01)
    >>> adjusted = processor.adjust_ps(ps, treatment)
    >>> print(np.round(adjusted, 3))
    [0.01 0.2  0.5  0.8  0.99]
    """

    _VALID_CALIBRATION_METHODS = {None, "isotonic"}

    def __init__(
        self,
        clipping_threshold: float = 1e-2,
        extreme_threshold: float = 1e-12,
        calibration_method: Optional[str] = None,
        cv_calibration: bool = False,
    ):
        self._clipping_threshold = clipping_threshold
        self._extreme_threshold = extreme_threshold
        self._calibration_method = calibration_method
        self._cv_calibration = cv_calibration

        self._validate_config()

    @classmethod
    def from_config(cls, config: PSProcessorConfig):
        """Create PSProcessor from PSProcessorConfig."""
        return cls(
            clipping_threshold=config.clipping_threshold,
            extreme_threshold=config.extreme_threshold,
            calibration_method=config.calibration_method,
            cv_calibration=config.cv_calibration,
        )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def clipping_threshold(self) -> float:
        """Get the clipping threshold."""
        return self._clipping_threshold

    @property
    def extreme_threshold(self) -> float:
        """Get the extreme threshold."""
        return self._extreme_threshold

    @property
    def calibration_method(self) -> Optional[str]:
        """Get the calibration method."""
        return self._calibration_method

    @property
    def cv_calibration(self) -> bool:
        """Get whether cross-validation calibration is used."""
        return self._cv_calibration

    # -------------------------------------------------------------------------
    # Core functionality
    # -------------------------------------------------------------------------
    def adjust_ps(
        self,
        propensity_scores: np.ndarray,
        treatment: np.ndarray,
        cv: Optional[Union[int, list]] = None,
        learner_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Adjust propensity scores via calibration and clipping.

        Parameters
        ----------
        propensity_scores : np.ndarray
            Raw propensity score predictions.
        treatment : np.ndarray
            Treatment assignments (1 for treated, 0 for control).
        cv : int or list, optional
            Cross-validation strategy for calibration. Used only if calibration is applied.
        learner_name : str, optional
            Name of the learner providing the propensity scores, used in warnings.

        Returns
        -------
        np.ndarray
            Clipped and validated propensity scores.
        """
        self._validate_propensity_scores(
            propensity_scores,
            learner_name,
        )
        self._validate_treatment(treatment)

        calibrated_ps = self._apply_calibration(propensity_scores, treatment, cv=cv)
        clipped_scores = np.clip(calibrated_ps, a_min=self.clipping_threshold, a_max=1 - self.clipping_threshold)

        return clipped_scores

    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------
    def _apply_calibration(
        self,
        propensity_scores: np.ndarray,
        treatment: np.ndarray,
        cv: Optional[Union[int, list]] = None,
    ) -> np.ndarray:
        """Apply calibration method to propensity scores if specified."""
        if self.calibration_method is None:
            calibrated_ps = propensity_scores
        elif self.calibration_method == "isotonic":
            calibration_model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

            if self.cv_calibration:
                calibrated_ps = cross_val_predict(
                    estimator=calibration_model, X=propensity_scores.reshape(-1, 1), y=treatment, cv=cv, method="predict"
                )
            else:
                calibration_model.fit(propensity_scores.reshape(-1, 1), treatment)
                calibrated_ps = calibration_model.predict(propensity_scores.reshape(-1, 1))
        else:
            # This point should never be reached due to prior validation
            raise ValueError(
                f"Unsupported calibration method: {self.calibration_method}. "
                f"Valid methods are: {self._VALID_CALIBRATION_METHODS}"
            )

        return calibrated_ps

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if not isinstance(self.clipping_threshold, float):
            raise TypeError("clipping_threshold must be a float.")
        if not (0 < self.clipping_threshold < 0.5):
            raise ValueError("clipping_threshold must be between 0 and 0.5.")

        if not (0 < self.extreme_threshold < 0.5):
            raise ValueError("extreme_threshold must be between 0 and 0.5.")

        if self.calibration_method not in self._VALID_CALIBRATION_METHODS:
            raise ValueError(f"calibration_method must be one of {self._VALID_CALIBRATION_METHODS}.")

        if not isinstance(self.cv_calibration, bool):
            raise TypeError("cv_calibration must be of bool type.")
        if self.cv_calibration and self.calibration_method is None:
            raise ValueError("cv_calibration=True requires a calibration_method.")

    def _validate_propensity_scores(
        self,
        preds: np.ndarray,
        learner_name: Optional[str] = None,
    ) -> None:
        """Validate if propensity predictions are valid."""
        learner_msg = f" from learner {learner_name}" if learner_name is not None else ""

        if not isinstance(preds, np.ndarray):
            raise TypeError(f"Propensity predictions {learner_msg} must be of type np.ndarray. " f"Type {type(preds)} found.")

        if preds.ndim != 1:
            raise ValueError(f"Propensity predictions {learner_msg} must be 1-dimensional. " f"Shape {preds.shape} found.")

        if any((preds < self.extreme_threshold) | (preds > 1 - self.extreme_threshold)):
            warnings.warn(
                f"Propensity predictions {learner_msg} " f"are close to zero or one (eps={self.extreme_threshold}).",
                UserWarning,
            )

    def _validate_treatment(self, treatment: np.ndarray) -> None:
        """Validate treatment vector."""
        if not isinstance(treatment, np.ndarray):
            raise TypeError(f"Treatment assignments must be of type np.ndarray. " f"Type {type(treatment)} found.")

        if treatment.ndim != 1:
            raise ValueError(f"Treatment assignments must be 1-dimensional. " f"Shape {treatment.shape} found.")

        binary_treat = type_of_target(treatment) == "binary"
        zero_one_treat = np.all((np.power(treatment, 2) - treatment) == 0)
        if not (binary_treat and zero_one_treat):
            raise ValueError("Treatment vector must be binary (0 and 1).")
