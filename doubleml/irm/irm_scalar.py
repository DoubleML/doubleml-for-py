"""
Interactive Regression Model (IRM) based on the new DoubleMLScalar hierarchy.
"""

from __future__ import annotations

from typing import ClassVar, Self

import numpy as np
from sklearn.base import clone
from sklearn.utils.multiclass import type_of_target

from ..data.base_data import DoubleMLData
from ..double_ml_linear_score import LinearScoreMixin
from ..utils._checks import _check_score, _check_weights
from ..utils._learner import LearnerSpec, predict_nuisance
from ..utils._propensity_score import _propensity_score_adjustment
from ..utils.propensity_score_processing import PSProcessor, PSProcessorConfig
from ..utils.resampling import DoubleMLResampling


class IRM(LinearScoreMixin):
    """Double machine learning for interactive regression models.

    Based on the DoubleMLScalar + LinearScoreMixin hierarchy.

    Parameters
    ----------
    obj_dml_data : DoubleMLData
        The data object providing the data and specifying the variables for the causal model.
        Must contain exactly one binary treatment variable with values 0 and 1.
    score : str
        The score function (``'ATE'`` or ``'ATTE'``).
        Default is ``'ATE'``.
    ml_g : estimator, optional
        A machine learner implementing ``fit()`` and ``predict()`` for the nuisance
        function :math:`g_0(D, X) = E[Y|X, D]`. Cloned to ``ml_g0`` and ``ml_g1``
        internally. For a binary outcome, a classifier implementing ``fit()`` and
        ``predict_proba()`` can also be specified.
    ml_m : classifier, optional
        A machine learner implementing ``fit()`` and ``predict_proba()`` for the
        nuisance function :math:`m_0(X) = E[D|X]`. Must be a classifier.
    normalize_ipw : bool
        Indicates whether the inverse probability weights are normalized.
        Default is ``False``.
    weights : array, dict or None
        Weights for each individual observation. If ``None``, uniform weights are used
        (corresponds to standard ATE). Can only be used with ``score='ATE'``.
        An array must have shape ``(n,)``. A dictionary must contain keys ``'weights'``
        and ``'weights_bar'``.
        Default is ``None``.
    ps_processor_config : PSProcessorConfig, optional
        Configuration for propensity score processing (clipping, calibration, etc.).
        Default is ``None`` (uses default clipping threshold of 0.01).

    Notes
    -----
    **Interactive regression (IRM)** models take the form

    .. math::

        Y = g_0(D, X) + U, & &\\mathbb{E}(U | X, D) = 0,

        D = m_0(X) + V, & &\\mathbb{E}(V | X) = 0,

    where the treatment variable is binary, :math:`D \\in \\lbrace 0,1 \\rbrace`.
    Target parameters of interest are the average treatment effect (ATE),

    .. math::

        \\theta_0 = \\mathbb{E}[g_0(1, X) - g_0(0, X)]

    and the average treatment effect of the treated (ATTE),

    .. math::

        \\theta_0 = \\mathbb{E}[g_0(1, X) - g_0(0, X) | D=1].
    """

    # Define learner specifications for IRM
    _LEARNER_SPECS: ClassVar[dict[str, LearnerSpec]] = {
        "ml_g0": LearnerSpec("ml_g0", allow_regressor=True, allow_classifier=True, binary_data_check="outcome"),
        "ml_g1": LearnerSpec("ml_g1", allow_regressor=True, allow_classifier=True, binary_data_check="outcome"),
        "ml_m": LearnerSpec("ml_m", allow_regressor=False, allow_classifier=True),
    }

    def __init__(
        self,
        obj_dml_data: DoubleMLData,
        score: str = "ATE",
        ml_g: object | None = None,
        ml_m: object | None = None,
        normalize_ipw: bool = False,
        weights: np.ndarray | dict | None = None,
        ps_processor_config: PSProcessorConfig | None = None,
    ):
        """
        Initialize IRM model.

        Parameters
        ----------
        obj_dml_data : DoubleMLData
            The data object. Must have exactly one binary treatment variable.
        score : str
            Score function (``'ATE'`` or ``'ATTE'``).
        ml_g : estimator, optional
            Learner for E[Y|X, D]. Cloned to ml_g0 and ml_g1.
        ml_m : classifier, optional
            Learner for E[D|X]. Must be a classifier.
        normalize_ipw : bool
            Whether to normalize inverse probability weights.
        weights : array, dict or None, optional
            Weights for weighted ATE.
        ps_processor_config : PSProcessorConfig, optional
            Configuration for propensity score processing.
        """
        # Validate data
        self._check_data(obj_dml_data)

        # Validate score
        valid_scores = ["ATE", "ATTE"]
        _check_score(score, valid_scores, allow_callable=False)

        super().__init__(
            obj_dml_data=obj_dml_data,
            score=score,
        )

        # Normalize IPW
        if not isinstance(normalize_ipw, bool):
            raise TypeError("Normalization indicator has to be boolean. " f"Object of type {str(type(normalize_ipw))} passed.")
        self._normalize_ipw = normalize_ipw

        # Propensity score processing
        if ps_processor_config is not None:
            self._ps_processor_config = ps_processor_config
            self._ps_processor = PSProcessor.from_config(ps_processor_config)
        else:
            self._ps_processor_config = PSProcessorConfig()
            self._ps_processor = PSProcessor.from_config(self._ps_processor_config)

        # Weights
        _check_weights(weights, score, obj_dml_data.n_obs, n_rep=1)
        self._initialize_weights(weights)

        # Set learners if provided
        if any(learner is not None for learner in [ml_g, ml_m]):
            self.set_learners(ml_g=ml_g, ml_m=ml_m)

    # ==================== Properties ====================

    @property
    def normalize_ipw(self) -> bool:
        """Indicates whether the inverse probability weights are normalized."""
        return self._normalize_ipw

    @property
    def ps_processor_config(self) -> PSProcessorConfig:
        """Configuration for propensity score processing."""
        return self._ps_processor_config

    @property
    def ps_processor(self) -> PSProcessor:
        """Propensity score processor."""
        return self._ps_processor

    @property
    def weights(self) -> dict:
        """Weights for weighted ATE/ATTE."""
        return self._weights

    @property
    def required_learners(self) -> list[str]:
        """Required learners for IRM: ml_g0, ml_g1, and ml_m."""
        return ["ml_g0", "ml_g1", "ml_m"]

    # ==================== Learner Management ====================

    def set_learners(
        self,
        ml_g: object | None = None,
        ml_g0: object | None = None,
        ml_g1: object | None = None,
        ml_m: object | None = None,
    ) -> Self:
        """
        Set the learners for nuisance estimation.

        Parameters
        ----------
        ml_g : estimator or None, optional
            A machine learner for the outcome regression :math:`g_0(D, X) = E[Y|X, D]`.
            Cloned to ``ml_g0`` and ``ml_g1`` if they are not explicitly set.
        ml_g0 : estimator or None, optional
            A machine learner for :math:`E[Y|X, D=0]`. Takes precedence over ``ml_g``.
        ml_g1 : estimator or None, optional
            A machine learner for :math:`E[Y|X, D=1]`. Takes precedence over ``ml_g``.
        ml_m : classifier or None, optional
            A machine learner for the propensity score :math:`m_0(X) = E[D|X]`.
            Must be a classifier with ``predict_proba()`` method.

        Returns
        -------
        self : IRM
            The estimator with learners set.
        """
        # ml_g convenience: clone to ml_g0/ml_g1 if not explicitly set
        if ml_g is not None:
            # Validate ml_g is an instance (not a class) before cloning
            if isinstance(ml_g, type):
                raise TypeError("Invalid learner provided for ml_g: provide an instance of a learner instead of a class.")
            if ml_g0 is None:
                ml_g0 = clone(ml_g)
            if ml_g1 is None:
                ml_g1 = clone(ml_g)

        # Register each learner
        for name, learner in [("ml_g0", ml_g0), ("ml_g1", ml_g1), ("ml_m", ml_m)]:
            if learner is not None:
                self._register_learner(name, learner)

        self._reset_fit_state()
        return self

    # ==================== Sample Splitting ====================

    def draw_sample_splitting(self, n_folds: int = 5, n_rep: int = 1) -> Self:
        """
        Draw stratified sample splitting for cross-fitting.

        Uses stratified K-fold splitting to ensure each fold contains both
        treatment groups (D=0 and D=1).

        Parameters
        ----------
        n_folds : int, optional
            Number of folds for cross-fitting. Default is 5.
        n_rep : int, optional
            Number of repetitions for sample splitting. Default is 1.

        Returns
        -------
        self : IRM
            The estimator with initialized sample splits.
        """
        if not isinstance(n_folds, int) or n_folds < 2:
            raise ValueError(f"n_folds must be an integer >= 2. Got {n_folds}.")
        if not isinstance(n_rep, int) or n_rep < 1:
            raise ValueError(f"n_rep must be an integer >= 1. Got {n_rep}.")

        self._n_folds = n_folds
        self._n_rep = n_rep

        # Create stratified resampler
        resampler = DoubleMLResampling(
            n_folds=n_folds,
            n_rep=n_rep,
            n_obs=self._n_obs,
            stratify=self._dml_data.d,
        )

        self._smpls = resampler.split_samples()
        return self

    # ==================== Nuisance Estimation ====================

    def _nuisance_est(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        i_rep: int,
        i_fold: int,
        external_predictions: dict[str, np.ndarray] | None = None,
    ) -> None:
        x = self._dml_data.x
        y = self._dml_data.y
        d = self._dml_data.d

        x_train, x_test = x[train_idx], x[test_idx]
        d_train = d[train_idx]

        # Check which learners have external predictions
        g0_external = external_predictions is not None and "ml_g0" in external_predictions
        g1_external = external_predictions is not None and "ml_g1" in external_predictions
        m_external = external_predictions is not None and "ml_m" in external_predictions

        # ml_g0: fit on d==0 subset of training data, predict on ALL test observations
        if not g0_external:
            train_d0 = train_idx[d[train_idx] == 0]
            ml_g0_info = self._learners["ml_g0"]
            ml_g0 = clone(ml_g0_info.learner)
            ml_g0.fit(x[train_d0], y[train_d0])
            self._predictions["ml_g0"][test_idx, i_rep] = predict_nuisance(ml_g0, x_test, ml_g0_info.is_classifier)

        # ml_g1: fit on d==1 subset of training data, predict on ALL test observations
        if not g1_external:
            train_d1 = train_idx[d[train_idx] == 1]
            ml_g1_info = self._learners["ml_g1"]
            ml_g1 = clone(ml_g1_info.learner)
            ml_g1.fit(x[train_d1], y[train_d1])
            self._predictions["ml_g1"][test_idx, i_rep] = predict_nuisance(ml_g1, x_test, ml_g1_info.is_classifier)

        # ml_m: fit on ALL training data, predict on test
        if not m_external:
            ml_m_info = self._learners["ml_m"]
            ml_m = clone(ml_m_info.learner)
            ml_m.fit(x_train, d_train)
            self._predictions["ml_m"][test_idx, i_rep] = predict_nuisance(ml_m, x_test, ml_m_info.is_classifier)

    # ==================== Score Elements ====================

    def _get_score_elements(self) -> dict[str, np.ndarray]:
        y = self._dml_data.y
        d = self._dml_data.d

        g_hat0 = self._predictions["ml_g0"]  # (n_obs, n_rep)
        g_hat1 = self._predictions["ml_g1"]  # (n_obs, n_rep)
        m_hat_raw = self._predictions["ml_m"]  # (n_obs, n_rep)

        # Apply PS processing per repetition
        m_hat = np.zeros_like(m_hat_raw)
        for i_rep in range(self.n_rep):
            m_hat[:, i_rep] = self._ps_processor.adjust_ps(m_hat_raw[:, i_rep], d, cv=self._smpls[i_rep], learner_name="ml_m")

        # Apply IPW normalization per repetition
        m_hat_adj = np.zeros_like(m_hat)
        for i_rep in range(self.n_rep):
            m_hat_adj[:, i_rep] = _propensity_score_adjustment(
                propensity_score=m_hat[:, i_rep],
                treatment_indicator=d,
                normalize_ipw=self.normalize_ipw,
            )

        # Residuals: (n_obs, n_rep)
        u_hat0 = y[:, np.newaxis] - g_hat0
        u_hat1 = y[:, np.newaxis] - g_hat1

        d_col = d[:, np.newaxis]  # (n_obs, 1) for broadcasting

        if self.score == "ATE" or self.score == "ATTE":
            weights, weights_bar = self._get_weights(m_hat_adj)

            psi_b = weights * (g_hat1 - g_hat0) + weights_bar * (
                np.divide(d_col * u_hat1, m_hat_adj) - np.divide((1.0 - d_col) * u_hat0, 1.0 - m_hat_adj)
            )
            psi_a = -1.0 * np.divide(weights, np.mean(weights, axis=0, keepdims=True))

        return {"psi_a": psi_a, "psi_b": psi_b}

    # ==================== Private Helpers ====================

    @staticmethod
    def _check_data(obj_dml_data: object) -> None:
        """Validate that the data is compatible with IRM."""
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError(
                f"The data must be of DoubleMLData type. " f"{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed."
            )
        if obj_dml_data.z_cols is not None:
            raise ValueError(
                "Incompatible data. " + " and ".join(obj_dml_data.z_cols) + " have been set as instrumental variable(s). "
                "To fit an interactive IV regression model use DoubleMLIIVM instead of IRM."
            )
        one_treat = obj_dml_data.n_treat == 1
        binary_treat = type_of_target(obj_dml_data.d) == "binary"
        zero_one_treat = np.all((np.power(obj_dml_data.d, 2) - obj_dml_data.d) == 0)
        if not (one_treat & binary_treat & zero_one_treat):
            raise ValueError(
                "Incompatible data. "
                "To fit an IRM model with DML "
                "exactly one binary variable with values 0 and 1 "
                "needs to be specified as treatment variable."
            )

    def _initialize_weights(self, weights: np.ndarray | dict | None) -> None:
        """Initialize weights storage."""
        if weights is None:
            weights = np.ones(self._dml_data.n_obs)
        if isinstance(weights, np.ndarray):
            self._weights = {"weights": weights}
        else:
            if not isinstance(weights, dict):
                raise TypeError(f"weights must be np.ndarray or dict, got {type(weights).__name__}")
            self._weights = weights

    def _get_weights(self, m_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute weights and weights_bar for score computation.

        Parameters
        ----------
        m_hat : np.ndarray
            Adjusted propensity scores, shape (n_obs, n_rep).

        Returns
        -------
        weights : np.ndarray
            Shape (n_obs, n_rep) or broadcastable.
        weights_bar : np.ndarray
            Shape (n_obs, n_rep) or broadcastable.
        """
        d = self._dml_data.d

        if self.score == "ATE":
            w = self._weights["weights"]
            weights = w[:, np.newaxis] * np.ones((1, self.n_rep))  # (n_obs, n_rep)
            if "weights_bar" in self._weights:
                # weights_bar has shape (n_obs, n_rep) already
                weights_bar = self._weights["weights_bar"]
            else:
                weights_bar = weights.copy()
        else:
            # ATTE (score validated in __init__)
            w = self._weights["weights"]
            subgroup = w * d
            subgroup_probability = np.mean(subgroup)
            weights = np.divide(subgroup, subgroup_probability)[:, np.newaxis] * np.ones((1, self.n_rep))

            # weights_bar depends on m_hat per repetition
            weights_bar = np.divide(m_hat * w[:, np.newaxis], subgroup_probability)

        return weights, weights_bar
