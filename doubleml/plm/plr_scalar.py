"""
Partially Linear Regression (PLR) model based on the new DoubleMLScalar hierarchy.
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar, Self

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict

from ..data.base_data import DoubleMLData
from ..double_ml_linear_score import LinearScoreMixin
from ..utils._checks import _check_binary_predictions, _check_finite_predictions, _check_is_propensity
from ..utils._learner import LearnerSpec, predict_nuisance


class PLR(LinearScoreMixin):
    """Double machine learning for partially linear regression models.

    Based on the DoubleMLScalar + LinearScoreMixin hierarchy.

    Parameters
    ----------
    obj_dml_data : DoubleMLData
        The data object providing the data and specifying the variables for the causal model.
    score : str
        The score function (``'partialling out'`` or ``'IV-type'``).
        Default is ``'partialling out'``.
    ml_l : estimator, optional
        Learner for E[Y|X]. Can be regressor or classifier.
    ml_m : estimator, optional
        Learner for E[D|X]. Can be regressor or classifier.
    ml_g : estimator, optional
        Learner for E[Y - D*theta|X]. Only for IV-type. Must be regressor.
    """

    # Define learner specifications for PLR
    _LEARNER_SPECS: ClassVar[dict[str, LearnerSpec]] = {
        "ml_l": LearnerSpec("ml_l", allow_regressor=True, allow_classifier=True, binary_data_check="outcome"),
        "ml_m": LearnerSpec("ml_m", allow_regressor=True, allow_classifier=True, binary_data_check="treatment"),
        "ml_g": LearnerSpec("ml_g", allow_regressor=True, allow_classifier=False),
    }

    def __init__(
        self,
        obj_dml_data: DoubleMLData,
        score: str = "partialling out",
        ml_l: object | None = None,
        ml_m: object | None = None,
        ml_g: object | None = None,
    ):
        """
        Initialize PLR model.

        Parameters
        ----------
        obj_dml_data : DoubleMLData
            The data object.
        score : str
            Score function ('partialling out' or 'IV-type').
        ml_l : estimator, optional
            Learner for E[Y|X]. Can be regressor or classifier.
        ml_m : estimator, optional
            Learner for E[D|X]. Can be regressor or classifier.
        ml_g : estimator, optional
            Learner for E[Y - D*theta|X]. Only for IV-type. Must be regressor.
        """
        # Validate data
        self._check_data(obj_dml_data)

        # Validate score
        valid_scores = ["partialling out", "IV-type"]
        if score not in valid_scores:
            raise ValueError(f"Invalid score '{score}'. Valid scores: {valid_scores}.")
        if score == "IV-type" and obj_dml_data.binary_outcome:
            raise ValueError("For score = 'IV-type', additive probability models (binary outcomes) are not supported.")

        super().__init__(
            obj_dml_data=obj_dml_data,
            score=score,
        )

        # Set learners if provided
        if any(learner is not None for learner in [ml_l, ml_m, ml_g]):
            self.set_learners(ml_l=ml_l, ml_m=ml_m, ml_g=ml_g)

    @property
    def required_learners(self) -> list[str]:
        """Required learners for current score."""
        names = ["ml_l", "ml_m"]
        if self.score == "IV-type":
            names.append("ml_g")
        return names

    def set_learners(
        self,
        ml_l: object | None = None,
        ml_m: object | None = None,
        ml_g: object | None = None,
    ) -> Self:
        """
        Set the learners for nuisance estimation.

        Parameters
        ----------
        ml_l : estimator or None, optional
            A machine learner implementing ``fit()`` and ``predict()`` for the nuisance
            function :math:`\\ell_0(X) = E[Y|X]`.
        ml_m : estimator or None, optional
            A machine learner implementing ``fit()`` and ``predict()`` for the nuisance
            function :math:`m_0(X) = E[D|X]`.
        ml_g : estimator or None, optional
            A machine learner implementing ``fit()`` and ``predict()`` for the nuisance
            function :math:`g_0(X) = E[Y - D\\theta_0|X]`.
            Only required for ``score='IV-type'``.

        Returns
        -------
        self : PLR
            The estimator with learners set.
        """
        for name, learner in [("ml_l", ml_l), ("ml_m", ml_m), ("ml_g", ml_g)]:
            if learner is None:
                continue
            if name not in self.required_learners:
                warnings.warn(f"Learner '{name}' not required for score='{self.score}', ignored.")
                continue
            self._register_learner(name, learner)

        # Warn when a classifier is used for ml_l with a binary outcome
        if ml_l is not None and "ml_l" in self._learners:
            if self._learners["ml_l"].is_classifier and self._dml_data.binary_outcome:
                warnings.warn(
                    f"The ml_l learner {str(ml_l)} was identified as classifier. " "Fitting an additive probability model.",
                    UserWarning,
                )

        # IV-type: clone ml_l to ml_g if only one provided
        self._handle_iv_cloning()
        self._reset_fit_state()
        return self

    def _handle_iv_cloning(self) -> None:
        """For IV-type score: clone ml_l to ml_g or vice versa if one is missing."""
        if self.score != "IV-type":
            return
        if "ml_g" not in self.required_learners:
            return

        has_l = "ml_l" in self._learners
        has_g = "ml_g" in self._learners

        if has_l and not has_g:
            warnings.warn("For score='IV-type', ml_g not set. Cloning ml_l to ml_g.")
            # Clone the learner and register with same info
            from ..utils._learner import LearnerInfo

            ml_l_info = self._learners["ml_l"]
            self._learners["ml_g"] = LearnerInfo(
                learner=clone(ml_l_info.learner),
                is_classifier=ml_l_info.is_classifier,
            )
        elif has_g and not has_l:
            warnings.warn("For score='IV-type', ml_l not set. Cloning ml_g to ml_l.")
            from ..utils._learner import LearnerInfo

            ml_g_info = self._learners["ml_g"]
            self._learners["ml_l"] = LearnerInfo(
                learner=clone(ml_g_info.learner),
                is_classifier=ml_g_info.is_classifier,
            )

    @staticmethod
    def _check_data(obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError(
                f"The data must be of DoubleMLData type. " f"{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed."
            )
        if obj_dml_data.z_cols is not None:
            raise ValueError(
                "Incompatible data. " + " and ".join(obj_dml_data.z_cols) + " have been set as instrumental variable(s). "
                "To fit a partially linear IV regression model use DoubleMLPLIV instead of DoubleMLPLR."
            )

    def _post_nuisance_checks(self) -> None:
        """Check predictions for validity after cross-fitting completes."""
        for i_rep in range(self.n_rep):
            # After full K-fold cross-fitting, all observations are test observations
            # in exactly one fold, so the full prediction array is populated.

            # Skip checks for learners with external predictions (not registered in _learners)
            if "ml_l" in self._learners:
                _check_finite_predictions(self._predictions["ml_l"][:, i_rep], self._learners["ml_l"].learner, "ml_l")
            if "ml_m" in self._learners:
                _check_finite_predictions(self._predictions["ml_m"][:, i_rep], self._learners["ml_m"].learner, "ml_m")

                # Propensity score range check when ml_m is a classifier
                if self._learners["ml_m"].is_classifier:
                    _check_is_propensity(
                        self._predictions["ml_m"][:, i_rep],
                        self._learners["ml_m"].learner,
                        "ml_m",
                    )

                # Binary predictions check for binary treatment
                if self._dml_data.binary_treats.all():
                    _check_binary_predictions(
                        self._predictions["ml_m"][:, i_rep],
                        self._learners["ml_m"].learner,
                        "ml_m",
                        self._dml_data.d_cols[0],
                    )

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
        y_train = y[train_idx]
        d_train = d[train_idx]

        # Check which learners have external predictions
        l_external = external_predictions is not None and "ml_l" in external_predictions
        m_external = external_predictions is not None and "ml_m" in external_predictions
        g_external = external_predictions is not None and "ml_g" in external_predictions

        # Fit and predict ml_l: E[Y|X]
        if not l_external:
            ml_l_info = self._learners["ml_l"]
            ml_l = clone(ml_l_info.learner)
            ml_l.fit(x_train, y_train)
            self._predictions["ml_l"][test_idx, i_rep] = predict_nuisance(ml_l, x_test, ml_l_info.is_classifier)

        # Fit and predict ml_m: E[D|X]
        if not m_external:
            ml_m_info = self._learners["ml_m"]
            ml_m = clone(ml_m_info.learner)
            ml_m.fit(x_train, d_train)
            self._predictions["ml_m"][test_idx, i_rep] = predict_nuisance(ml_m, x_test, ml_m_info.is_classifier)

        # For IV-type: fit ml_g after last fold when all ml_l/ml_m predictions are available
        is_last_fold = i_fold == self.n_folds - 1
        if is_last_fold and self.score == "IV-type" and not g_external:
            # If ml_g not explicitly set, clone ml_l (already handled in _handle_iv_cloning)
            if "ml_g" not in self._learners:
                warnings.warn("For score = 'IV-type', learners ml_l and ml_g should be specified. Set ml_g = clone(ml_l).")
                from ..utils._learner import LearnerInfo

                ml_l_info = self._learners["ml_l"]
                self._learners["ml_g"] = LearnerInfo(
                    learner=clone(ml_l_info.learner),
                    is_classifier=ml_l_info.is_classifier,
                )

            # Compute initial theta from full cross-fitted predictions
            l_hat = self._predictions["ml_l"][:, i_rep]
            m_hat = self._predictions["ml_m"][:, i_rep]
            psi_a = -(d - m_hat) * (d - m_hat)
            psi_b = (d - m_hat) * (y - l_hat)
            theta_initial = -np.nanmean(psi_b) / np.nanmean(psi_a)

            # Second pass: fit ml_g with cross-fitting across all folds
            ml_g_info = self._learners["ml_g"]
            for j_fold in range(self.n_folds):
                train_j, test_j = self._smpls[i_rep][j_fold]
                ml_g = clone(ml_g_info.learner)
                ml_g.fit(x[train_j], y[train_j] - theta_initial * d[train_j])
                self._predictions["ml_g"][test_j, i_rep] = predict_nuisance(ml_g, x[test_j], ml_g_info.is_classifier)

    def _get_tuning_data(
        self,
        learner_name: str,
        partial_results: dict[str, Any],
        cv: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return ``(y_target, x)`` for tuning the given PLR learner.

        Parameters
        ----------
        learner_name : str
            Learner to tune: ``'ml_l'``, ``'ml_m'``, or ``'ml_g'``.
        partial_results : dict
            Already-tuned DMLOptunaResult objects, keyed by learner name.
            Used for 2-stage ``ml_g`` tuning: applies the best params from
            ``ml_l`` and ``ml_m`` when computing the initial theta estimate.
            If ``ml_l`` or ``ml_m`` were not tuned in this call, their current
            (untuned) learner params are used as a fallback.
        cv : cross-validator
            Cross-validation splitter, already resolved in :meth:`tune_ml_models`.

        Returns
        -------
        y_target : np.ndarray
            Target array for the learner.
        x : np.ndarray
            Feature matrix.

        Raises
        ------
        ValueError
            If ``learner_name`` is not a valid PLR learner name.
        """
        y = self._dml_data.y
        d = self._dml_data.d
        x = self._dml_data.x

        if learner_name == "ml_l":
            return y, x
        if learner_name == "ml_m":
            return d, x
        if learner_name == "ml_g":
            # 2-stage: compute initial theta via cross-validated ml_l/ml_m predictions.
            # Apply tuned params if available, otherwise use the current learner params.
            if "ml_l" not in self._learners or "ml_m" not in self._learners:
                raise ValueError(
                    "Tuning 'ml_g' requires 'ml_l' and 'ml_m' to be registered. "
                    "Call set_learners(ml_l=..., ml_m=...) before tuning 'ml_g'."
                )
            l_info = self._learners["ml_l"]
            m_info = self._learners["ml_m"]

            l_est = clone(l_info.learner)
            if "ml_l" in partial_results:
                l_est.set_params(**partial_results["ml_l"].best_params)

            m_est = clone(m_info.learner)
            if "ml_m" in partial_results:
                m_est.set_params(**partial_results["ml_m"].best_params)

            if l_info.is_classifier:
                l_hat = cross_val_predict(l_est, x, y, cv=cv, method="predict_proba")[:, 1]
            else:
                l_hat = cross_val_predict(l_est, x, y, cv=cv)

            if m_info.is_classifier:
                m_hat = cross_val_predict(m_est, x, d, cv=cv, method="predict_proba")[:, 1]
            else:
                m_hat = cross_val_predict(m_est, x, d, cv=cv)

            psi_a = -((d - m_hat) ** 2)
            psi_b = (d - m_hat) * (y - l_hat)
            theta_initial = -np.nanmean(psi_b) / np.nanmean(psi_a)
            return y - theta_initial * d, x

        raise ValueError(f"Unknown learner '{learner_name}' for PLR.")

    def _get_nuisance_targets(self) -> dict[str, np.ndarray | None]:
        """Return target arrays for nuisance loss evaluation.

        Returns y for ml_l, d for ml_m. For IV-type score, ml_g target is None because
        the adjusted outcome y - θ·d depends on the estimated parameter and varies per
        repetition, so it cannot be recovered post-fit.
        """
        y = self._dml_data.y
        d = self._dml_data.d
        targets: dict[str, np.ndarray | None] = {
            "ml_l": np.tile(y[:, np.newaxis], (1, self.n_rep)),
            "ml_m": np.tile(d[:, np.newaxis], (1, self.n_rep)),
        }
        if "ml_g" in self.required_learners:
            targets["ml_g"] = None
        return targets

    def _get_score_elements(self) -> dict[str, np.ndarray]:
        y = self._dml_data.y
        d = self._dml_data.d

        m_hat = self._predictions["ml_m"]  # (n_obs, n_rep)
        v_hat = d[:, np.newaxis] - m_hat  # (n_obs, n_rep)

        if self.score == "partialling out":
            l_hat = self._predictions["ml_l"]
            u_hat = y[:, np.newaxis] - l_hat
            psi_a = -v_hat * v_hat
            psi_b = v_hat * u_hat
        else:
            assert self.score == "IV-type"
            g_hat = self._predictions["ml_g"]
            psi_a = -v_hat * d[:, np.newaxis]
            psi_b = v_hat * (y[:, np.newaxis] - g_hat)

        return {"psi_a": psi_a, "psi_b": psi_b}

    def _sensitivity_element_est(self) -> dict[str, np.ndarray] | None:
        """
        Compute PLR sensitivity elements vectorized over all repetitions.

        Computes sigma2 (outcome residual variance), nu2 (inverse of treatment
        residual variance), their influence functions, and the Riesz representer.
        Handles both ``'partialling out'`` and ``'IV-type'`` scores.

        Returns
        -------
        dict[str, np.ndarray] or None
            Dictionary with keys ``'sigma2'``, ``'nu2'`` (shape ``(1, 1, n_rep)``),
            ``'psi_sigma2'``, ``'psi_nu2'``, ``'riesz_rep'`` (shape ``(n_obs, 1, n_rep)``).
            Returns ``None`` for callable scores (no standard Riesz representer).
        """
        if callable(self.score):
            return None

        y = self._dml_data.y  # (n_obs,)
        d = self._dml_data.d  # (n_obs,)
        m_hat = self._predictions["ml_m"]  # (n_obs, n_rep)
        theta = self._all_thetas  # (1, n_rep) — broadcasts with (n_obs, n_rep)

        treatment_residual = d[:, np.newaxis] - m_hat  # (n_obs, n_rep)

        if self.score == "partialling out":
            l_hat = self._predictions["ml_l"]  # (n_obs, n_rep)
            sigma2_score = (y[:, np.newaxis] - l_hat - theta * treatment_residual) ** 2
        else:  # "IV-type"
            g_hat = self._predictions["ml_g"]  # (n_obs, n_rep)
            sigma2_score = (y[:, np.newaxis] - g_hat - theta * d[:, np.newaxis]) ** 2

        # sigma2: mean across observations, reshaped to (1, 1, n_rep)
        sigma2_mean = np.mean(sigma2_score, axis=0)  # (n_rep,)
        psi_sigma2 = sigma2_score - sigma2_mean[np.newaxis, :]  # (n_obs, n_rep)
        sigma2 = sigma2_mean[np.newaxis, np.newaxis, :]  # (1, 1, n_rep)
        psi_sigma2 = psi_sigma2[:, np.newaxis, :]  # (n_obs, 1, n_rep)

        # nu2 = 1 / E[(d - m_hat)^2], reshaped to (1, 1, n_rep)
        tr_sq_mean = np.mean(treatment_residual**2, axis=0)  # (n_rep,)
        nu2_val = 1.0 / tr_sq_mean  # (n_rep,)
        psi_nu2 = nu2_val[np.newaxis, :] - treatment_residual**2 * nu2_val[np.newaxis, :] ** 2  # (n_obs, n_rep)
        nu2 = nu2_val[np.newaxis, np.newaxis, :]  # (1, 1, n_rep)
        psi_nu2 = psi_nu2[:, np.newaxis, :]  # (n_obs, 1, n_rep)

        # Riesz representer: (d - m_hat) * nu2
        rr = (treatment_residual * nu2_val[np.newaxis, :])[:, np.newaxis, :]  # (n_obs, 1, n_rep)

        return {
            "sigma2": sigma2,
            "nu2": nu2,
            "psi_sigma2": psi_sigma2,
            "psi_nu2": psi_nu2,
            "riesz_rep": rr,
        }
