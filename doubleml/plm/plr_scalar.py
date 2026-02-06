"""
Partially Linear Regression (PLR) model based on the new DoubleMLScalar hierarchy.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Self

import numpy as np
from sklearn.base import clone

from ..data.base_data import DoubleMLData
from ..double_ml_linear_score import LinearScoreMixin
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
    _LEARNER_SPECS: Dict[str, LearnerSpec] = {
        "ml_l": LearnerSpec("ml_l", allow_regressor=True, allow_classifier=True, binary_data_check="outcome"),
        "ml_m": LearnerSpec("ml_m", allow_regressor=True, allow_classifier=True, binary_data_check="treatment"),
        "ml_g": LearnerSpec("ml_g", allow_regressor=True, allow_classifier=False),
    }

    def __init__(
        self,
        obj_dml_data: DoubleMLData,
        score: str = "partialling out",
        ml_l: Optional[object] = None,
        ml_m: Optional[object] = None,
        ml_g: Optional[object] = None,
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

        super().__init__(
            obj_dml_data=obj_dml_data,
            score=score,
        )

        # Set learners if provided
        if any(learner is not None for learner in [ml_l, ml_m, ml_g]):
            self.set_learners(ml_l=ml_l, ml_m=ml_m, ml_g=ml_g)

    @property
    def required_learners(self) -> List[str]:
        """Required learners for current score."""
        names = ["ml_l", "ml_m"]
        if self.score == "IV-type":
            names.append("ml_g")
        return names

    def set_learners(
        self,
        ml_l: Optional[object] = None,
        ml_m: Optional[object] = None,
        ml_g: Optional[object] = None,
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

        # IV-type: clone ml_l to ml_g if only one provided
        self._handle_iv_cloning()
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

    def _nuisance_est(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        i_rep: int,
        i_fold: int,
        external_predictions: Optional[Dict[str, np.ndarray]] = None,
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

    def _get_score_elements(self) -> Dict[str, np.ndarray]:
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
