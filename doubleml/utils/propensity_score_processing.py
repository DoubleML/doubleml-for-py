import warnings
from typing import Any, Dict, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict
from sklearn.utils.multiclass import type_of_target


class PropensityScoreProcessor:
    """
    Processor for propensity score calibration, clipping, and validation.

    Parameters
    ----------
    clipping_threshold : float, default=1e-2
        Threshold used for clipping propensity scores.
    warn_extreme_values : bool, default=True
        Whether to warn about extreme propensity score values.
    extreme_threshold : float, default=0.05
        Threshold for extreme value warnings.
    warning_proportion : float, default=0.1
        Proportion threshold for triggering extreme value warnings.

    Examples
    --------
    >>> import numpy as np
    >>> from doubleml.utils import PropensityScoreProcessor
    >>> ps_scores = np.array([0.001, 0.2, 0.5, 0.8, 0.999])
    >>> treatment = np.array([0, 1, 1, 0, 1])
    >>> processor = PropensityScoreProcessor(clipping_threshold=0.01)
    >>> adj_scores = processor.adjust(ps_scores, treatment)
    >>> print(adj_scores)
    [0.01 0.2  0.5  0.8  0.99]
    """

    _DEFAULT_CONFIG: Dict[str, Any] = {
        "clipping_threshold": 1e-2,
        "extreme_threshold": 1e-12,
        "calibration_method": None,
        "cv_calibration": False,
    }

    _VALID_CALIBRATION_METHODS = {None, "isotonic"}

    def __init__(self, **config: Any) -> None:

        unknown_params = set(config.keys()) - set(self._DEFAULT_CONFIG.keys())
        if unknown_params:
            raise ValueError(f"Unknown parameters: {unknown_params}")

        updated_config = {**self._DEFAULT_CONFIG, **config}
        self._validate_config(updated_config)
        self._config = updated_config

    # -------------------------------------------------------------------------
    # Configuration methods
    # -------------------------------------------------------------------------
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration parameters."""

        clipping_threshold = config["clipping_threshold"]
        if not isinstance(clipping_threshold, float):
            raise TypeError("clipping_threshold must be of float type. " f"Object of type {type(clipping_threshold)} passed.")
        if (clipping_threshold <= 0) or (clipping_threshold >= 0.5):
            raise ValueError(f"clipping_threshold must be between 0 and 0.5. " f"{clipping_threshold} was passed.")

        if not (0 < config["extreme_threshold"] < 0.5):
            raise ValueError("extreme_threshold must be between 0 and 0.5.")

        calibration_method = config["calibration_method"]
        if calibration_method not in self._VALID_CALIBRATION_METHODS:
            raise ValueError(
                f"calibration_method must be one of {self._VALID_CALIBRATION_METHODS}. " f"Got {calibration_method}."
            )

        if not isinstance(config["cv_calibration"], bool):
            raise TypeError("cv_calibration must be of bool type.")
        if config["cv_calibration"] and config["calibration_method"] is None:
            raise ValueError("cv_calibration can only be used with a calibration_method.")

    @property
    def clipping_threshold(self) -> float:
        """Get the clipping threshold."""
        return self._config["clipping_threshold"]

    @property
    def extreme_threshold(self) -> float:
        """Get the extreme threshold."""
        return self._config["extreme_threshold"]

    @property
    def calibration_method(self) -> Optional[str]:
        """Get the calibration method."""
        return self._config["calibration_method"]

    @property
    def cv_calibration(self) -> bool:
        """Get whether cross-validation calibration is used."""
        return self._config["cv_calibration"]

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Return the default configuration dictionary."""
        return cls._DEFAULT_CONFIG.copy()

    def get_config(self) -> Dict[str, Any]:
        """Return a copy of the current configuration dictionary."""
        return self._config.copy()

    def update_config(self, **new_config: Any) -> None:
        """
        Update configuration parameters.

        Validates the new configuration before applying changes to ensure
        the object remains in a consistent state.
        """

        unknown_params = set(new_config.keys()) - set(self._DEFAULT_CONFIG.keys())
        if unknown_params:
            raise ValueError(f"Unknown parameters: {unknown_params}")

        updated_config = {**self._config, **new_config}
        self._validate_config(updated_config)
        self._config = updated_config

    # -------------------------------------------------------------------------
    # Core functionality
    # -------------------------------------------------------------------------
    def adjust(
        self,
        propensity_scores: np.ndarray,
        treatment: np.ndarray,
        cv: Optional[int | list] = None,
        learner_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Adjust propensity scores via validation, clipping, and warnings.

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

        if self.cv_calibration:
            cv = cv
        else:
            cv = None
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
        cv: Optional[int | list] = None,
    ) -> np.ndarray:
        """Apply calibration method to propensity scores if specified."""
        if self.calibration_method is None:
            calibrated_ps = propensity_scores
        elif self.calibration_method == "isotonic":
            calibration_model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

            if cv is None:
                calibration_model.fit(propensity_scores.reshape(-1, 1), treatment)
                calibrated_ps = calibration_model.predict(propensity_scores.reshape(-1, 1))
            else:
                calibrated_ps = cross_val_predict(
                    estimator=calibration_model, X=propensity_scores.reshape(-1, 1), y=treatment, cv=cv, method="predict"
                )

        else:
            # This point should never be reached due to prior validation
            raise ValueError(
                f"Unsupported calibration method: {self.calibration_method}. "
                f"Valid methods are: {self._VALID_CALIBRATION_METHODS}"
            )

        return calibrated_ps

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

    # -------------------------------------------------------------------------
    # Representations
    # -------------------------------------------------------------------------
    def __repr__(self) -> str:
        config_str = ", ".join([f"{k}={v}" for k, v in sorted(self._config.items())])
        return f"{self.__class__.__name__}({config_str})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PropensityScoreProcessor):
            return False
        return self._config == other._config
