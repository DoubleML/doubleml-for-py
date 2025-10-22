import warnings
from typing import Any, Dict, Optional

import numpy as np


class PropensityScoreProcessor:
    """
    Processor for propensity score validation, clipping, and warnings.

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
    >>> raw_scores = np.array([0.001, 0.2, 0.5, 0.8, 0.999])
    >>> processor = PropensityScoreProcessor(clipping_threshold=0.01)
    >>> clipped_scores = processor.adjust(raw_scores)
    >>> print(clipped_scores)
    [0.01 0.2  0.5  0.8  0.99]
    """

    _DEFAULT_CONFIG: Dict[str, Any] = {
        "clipping_threshold": 1e-2,
        "extreme_threshold": 1e-12,
    }

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

    @property
    def clipping_threshold(self) -> float:
        """Get the clipping threshold."""
        return self._config["clipping_threshold"]

    @property
    def extreme_threshold(self) -> float:
        """Get the extreme threshold."""
        return self._config["extreme_threshold"]

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
    def adjust(self, propensity_scores: np.ndarray, learner_name: Optional[str] = None) -> np.ndarray:
        """
        Adjust propensity scores via validation, clipping, and warnings.

        Parameters
        ----------
        propensity_scores : array-like
            Raw propensity score predictions.
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
        clipped_scores = np.clip(propensity_scores, a_min=self.clipping_threshold, a_max=1 - self.clipping_threshold)

        return clipped_scores

    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------

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
