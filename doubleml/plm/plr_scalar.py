"""
Partially Linear Regression (PLR) model based on the new DoubleMLScalar hierarchy.
"""

import warnings

import numpy as np
from sklearn.base import clone

from ..data.base_data import DoubleMLData
from ..double_ml_linear_score import LinearScoreMixin


class PLR(LinearScoreMixin):
    """Double machine learning for partially linear regression models.

    Based on the DoubleMLScalar + LinearScoreMixin hierarchy.

    Parameters
    ----------
    obj_dml_data : DoubleMLData
        The data object providing the data and specifying the variables for the causal model.
    ml_l : estimator
        A machine learner implementing ``fit()`` and ``predict()`` for the nuisance
        function :math:`\\ell_0(X) = E[Y|X]`.
    ml_m : estimator
        A machine learner implementing ``fit()`` and ``predict()`` for the nuisance
        function :math:`m_0(X) = E[D|X]`.
    ml_g : estimator, optional
        A machine learner implementing ``fit()`` and ``predict()`` for the nuisance
        function :math:`g_0(X) = E[Y - D\\theta_0|X]`.
        Only required for ``score='IV-type'``.
    score : str, optional
        The score function (``'partialling out'`` or ``'IV-type'``).
        Default is ``'partialling out'``.
    """

    def __init__(
        self,
        obj_dml_data,
        ml_l,
        ml_m,
        ml_g=None,
        score="partialling out",
    ):
        # Validate data
        self._check_data(obj_dml_data)

        # Validate score
        valid_scores = ["partialling out", "IV-type"]
        if score not in valid_scores:
            raise ValueError(f"Invalid score '{score}'. Valid scores: {valid_scores}.")

        # Store learners
        self._learner = {"ml_l": clone(ml_l), "ml_m": clone(ml_m)}

        if ml_g is not None:
            if score == "IV-type":
                self._learner["ml_g"] = clone(ml_g)
            else:
                warnings.warn(
                    "A learner ml_g has been provided for score = 'partialling out' but will be ignored. "
                    "A learner ml_g is not required for estimation."
                )
        elif score == "IV-type":
            warnings.warn("For score = 'IV-type', learners ml_l and ml_g should be specified. Set ml_g = clone(ml_l).")
            self._learner["ml_g"] = clone(ml_l)

        super().__init__(
            obj_dml_data=obj_dml_data,
            score=score,
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

    def _initialize_predictions_dict(self):
        n_obs = self._n_obs
        n_rep = self.n_rep
        preds = {
            "ml_l": np.full((n_obs, n_rep), np.nan),
            "ml_m": np.full((n_obs, n_rep), np.nan),
        }
        if "ml_g" in self._learner:
            preds["ml_g"] = np.full((n_obs, n_rep), np.nan)
        return preds

    def _nuisance_est(self, train_idx, test_idx, i_rep, i_fold, external_predictions=None):
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
            ml_l = clone(self._learner["ml_l"])
            ml_l.fit(x_train, y_train)
            self._predictions["ml_l"][test_idx, i_rep] = ml_l.predict(x_test)

        # Fit and predict ml_m: E[D|X]
        if not m_external:
            ml_m = clone(self._learner["ml_m"])
            ml_m.fit(x_train, d_train)
            self._predictions["ml_m"][test_idx, i_rep] = ml_m.predict(x_test)

        # For IV-type: fit ml_g after last fold when all ml_l/ml_m predictions are available
        is_last_fold = i_fold == self.n_folds - 1
        if is_last_fold and "ml_g" in self._learner and "ml_g" in self._predictions and not g_external:
            # Compute initial theta from full cross-fitted predictions
            l_hat = self._predictions["ml_l"][:, i_rep]
            m_hat = self._predictions["ml_m"][:, i_rep]
            psi_a = -(d - m_hat) * (d - m_hat)
            psi_b = (d - m_hat) * (y - l_hat)
            theta_initial = -np.nanmean(psi_b) / np.nanmean(psi_a)

            # Second pass: fit ml_g with cross-fitting across all folds
            for j_fold in range(self.n_folds):
                train_j, test_j = self._smpls[i_rep][j_fold]
                ml_g = clone(self._learner["ml_g"])
                ml_g.fit(x[train_j], y[train_j] - theta_initial * d[train_j])
                self._predictions["ml_g"][test_j, i_rep] = ml_g.predict(x[test_j])

    def _get_score_elements(self):
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
