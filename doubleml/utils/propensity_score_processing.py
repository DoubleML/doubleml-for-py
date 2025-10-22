import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from doubleml.utils._checks import _check_is_propensity


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
    >>> processor = PropensityScoreProcessor(clipping_threshold=0.01)
    >>> clipped_scores = processor.adjust(raw_scores)
    """

    _DEFAULT_CONFIG: Dict[str, Any] = {
        "clipping_threshold": 1e-2,
        "warn_extreme_values": True,
        "extreme_threshold": 0.05,
        "warning_proportion": 0.1,
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

        if not isinstance(config["warn_extreme_values"], bool):
            raise TypeError("warn_extreme_values must be boolean.")

        if not (0 < config["extreme_threshold"] < 0.5):
            raise ValueError("extreme_threshold must be between 0 and 0.5.")

        if not isinstance(config["warning_proportion"], float):
            raise TypeError(
                "warning_proportion must be of float type. " f"Object of type {type(config['warning_proportion'])} passed."
            )
        if not (0 < config["warning_proportion"] < 1):
            raise ValueError("warning_proportion must be between 0 and 1.")

    @property
    def clipping_threshold(self) -> float:
        """Get the clipping threshold."""
        return self._config["clipping_threshold"]

    @property
    def warn_extreme_values(self) -> bool:
        """Get the warn extreme values setting."""
        return self._config["warn_extreme_values"]

    @property
    def extreme_threshold(self) -> float:
        """Get the extreme threshold."""
        return self._config["extreme_threshold"]

    @property
    def warning_proportion(self) -> float:
        """Get the warning proportion."""
        return self._config["warning_proportion"]

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
        learner_name: str = "ml_m",
        smpls: Optional[List[Any]] = None,
    ) -> np.ndarray:
        """
        Adjust propensity scores via validation, clipping, and warnings.

        Parameters
        ----------
        propensity_scores : array-like
            Raw propensity score predictions.
        learner_name : str, default="ml_m"
            Name of the learner for error messages.
        smpls : list, optional
            Sample splits for validation.

        Returns
        -------
        np.ndarray
            Clipped and validated propensity scores.
        """
        # Validation
        _check_is_propensity(
            propensity_scores,
            learner_name,
            learner_name,
            smpls,
            eps=1e-12,
        )

        # Warnings for extreme values
        if self.warn_extreme_values:
            self._warn_extreme_values(propensity_scores)

        # Clipping
        clipped_scores = np.clip(propensity_scores, a_min=self.clipping_threshold, a_max=1 - self.clipping_threshold)

        return np.asarray(clipped_scores)

    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------
    def _warn_extreme_values(self, propensity_scores: np.ndarray) -> None:
        """Emit warnings for extreme or clipped propensity scores."""
        min_prop = np.min(propensity_scores)
        max_prop = np.max(propensity_scores)

        extreme_low = np.mean(propensity_scores < self.extreme_threshold)
        extreme_high = np.mean(propensity_scores > (1 - self.extreme_threshold))

        if extreme_low > self.warning_proportion:
            warnings.warn(
                f"Large proportion ({extreme_low:.1%}) of propensity scores "
                f"below {self.extreme_threshold}. This may indicate poor overlap. "
                f"Consider adjusting the model or increasing clipping_threshold "
                f"(current: {self.clipping_threshold}).",
                UserWarning,
            )

        if extreme_high > self.warning_proportion:
            warnings.warn(
                f"Large proportion ({extreme_high:.1%}) of propensity scores "
                f"above {1 - self.extreme_threshold}. This may indicate poor overlap. "
                f"Consider adjusting the model or increasing clipping_threshold "
                f"(current: {self.clipping_threshold}).",
                UserWarning,
            )

        if min_prop <= self.clipping_threshold:
            warnings.warn(
                f"Minimum propensity score ({min_prop:.6f}) is at or below "
                f"clipping threshold ({self.clipping_threshold}). "
                f"Some observations may be heavily clipped.",
                UserWarning,
            )

        if max_prop >= (1 - self.clipping_threshold):
            warnings.warn(
                f"Maximum propensity score ({max_prop:.6f}) is at or above "
                f"clipping threshold ({1 - self.clipping_threshold}). "
                f"Some observations may be heavily clipped.",
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
