import numpy as np
from sklearn.utils.multiclass import type_of_target
from sklearn.base import clone
from sklearn.utils import check_X_y
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..double_ml import DoubleML
from ..double_ml_score_mixins import NonLinearScoreMixin
from ..double_ml_data import DoubleMLData

from ..utils._estimation import (
    _dml_cv_predict,
    _trimm,
    _predict_zero_one_propensity,
    _cond_targets,
    _get_bracket_guess,
    _default_kde,
    _normalize_ipw,
    _dml_tune,
    _solve_ipw_score,
)
from ..utils._checks import _check_score, _check_trimming, _check_zero_one_treatment, _check_treatment, _check_quantile


class DoubleMLLPQ(NonLinearScoreMixin, DoubleML):
    """Double machine learning for local potential quantiles

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    ml_g : classifier implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance elements which depend on priliminary estimation.

    ml_m : classifier implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the treatment propensity nuisance functions.

    treatment : int
        Binary treatment indicator. Has to be either ``0`` or ``1``. Determines the potential outcome to be considered.
        Default is ``1``.

    quantile : float
        Quantile of the potential outcome. Has to be between ``0`` and ``1``.
        Default is ``0.5``.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitons for the sample splitting.
        Default is ``1``.

    score : str
        A str (``'PQ'`` is the only choice) specifying the score function
        for potential quantiles.
        Default is ``'PQ'``.

    normalize_ipw : bool
        Indicates whether the inverse probability weights are normalized.
        Default is ``True``.

    kde : callable or None
        A callable object / function with signature ``deriv = kde(u, weights)`` for weighted kernel density estimation.
        Here ``deriv`` should evaluate the density in ``0``.
        Default is ``'None'``, which uses :py:class:`statsmodels.nonparametric.kde.KDEUnivariate` with a
        gaussian kernel and silverman for bandwidth determination.

    trimming_rule : str
        A str (``'truncate'`` is the only choice) specifying the trimming approach.
        Default is ``'truncate'``.

    trimming_threshold : float
        The threshold used for trimming.
        Default is ``1e-2``.

    draw_sample_splitting : bool
        Indicates whether the sample splitting should be drawn during initialization of the object.
        Default is ``True``.

    Examples
    --------
    >>> import numpy as np
    >>> import doubleml as dml
    >>> from doubleml.datasets import make_iivm_data
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> np.random.seed(3141)
    >>> ml_g = RandomForestClassifier(n_estimators=100, max_features=20, max_depth=10, min_samples_leaf=2)
    >>> ml_m = RandomForestClassifier(n_estimators=100, max_features=20, max_depth=10, min_samples_leaf=2)
    >>> data = make_iivm_data(theta=0.5, n_obs=1000, dim_x=20, return_type='DataFrame')
    >>> obj_dml_data = dml.DoubleMLData(data, 'y', 'd', z_cols='z')
    >>> dml_lpq_obj = dml.DoubleMLLPQ(obj_dml_data, ml_g, ml_m, treatment=1, quantile=0.5)
    >>> dml_lpq_obj.fit().summary
           coef   std err         t    P>|t|    2.5 %    97.5 %
    d  0.217244  0.636453  0.341336  0.73285 -1.03018  1.464668
    """

    def __init__(
        self,
        obj_dml_data,
        ml_g,
        ml_m,
        treatment=1,
        quantile=0.5,
        n_folds=5,
        n_rep=1,
        score="LPQ",
        normalize_ipw=True,
        kde=None,
        trimming_rule="truncate",
        trimming_threshold=1e-2,
        draw_sample_splitting=True,
    ):
        super().__init__(obj_dml_data,
                         n_folds,
                         n_rep, score,
                         draw_sample_splitting)

        self._quantile = quantile
        self._treatment = treatment
        if kde is None:
            self._kde = _default_kde
        else:
            if not callable(kde):
                raise TypeError("kde should be either a callable or None. " "%r was passed." % kde)
            self._kde = kde
        self._normalize_ipw = normalize_ipw

        self._check_data(self._dml_data)

        valid_score = ["LPQ"]
        _check_score(self.score, valid_score, allow_callable=False)
        _check_quantile(self.quantile)
        _check_treatment(self.treatment)

        if not isinstance(self.normalize_ipw, bool):
            raise TypeError(
                "Normalization indicator has to be boolean. " + f"Object of type {str(type(self.normalize_ipw))} passed."
            )

        # initialize starting values and bounds
        self._coef_bounds = (self._dml_data.y.min(), self._dml_data.y.max())
        self._coef_start_val = np.quantile(self._dml_data.y[self._dml_data.d == self.treatment], self.quantile)

        # set stratication for resampling
        self._strata = self._dml_data.d.reshape(-1, 1) + 2 * self._dml_data.z.reshape(-1, 1)
        if draw_sample_splitting:
            self.draw_sample_splitting()

        self._external_predictions_implemented = True

        # initialize and check trimming
        self._trimming_rule = trimming_rule
        self._trimming_threshold = trimming_threshold
        _check_trimming(self._trimming_rule, self._trimming_threshold)

        _ = self._check_learner(ml_g, "ml_g", regressor=False, classifier=True)
        _ = self._check_learner(ml_m, "ml_m", regressor=False, classifier=True)
        self._learner = {
            "ml_m_z": clone(ml_m),
            "ml_g_du_z0": clone(ml_g),
            "ml_g_du_z1": clone(ml_g),
            "ml_m_d_z0": clone(ml_m),
            "ml_m_d_z1": clone(ml_m),
        }
        self._predict_method = {
            "ml_m_z": "predict_proba",
            "ml_g_du_z0": "predict_proba",
            "ml_g_du_z1": "predict_proba",
            "ml_m_d_z0": "predict_proba",
            "ml_m_d_z1": "predict_proba",
        }

        self._initialize_ml_nuisance_params()

    @property
    def quantile(self):
        """
        Quantile for potential outcome.
        """
        return self._quantile

    @property
    def treatment(self):
        """
        Treatment indicator for potential outcome.
        """
        return self._treatment

    @property
    def kde(self):
        """
        The kernel density estimation of the derivative.
        """
        return self._kde

    @property
    def normalize_ipw(self):
        """
        Indicates whether the inverse probability weights are normalized.
        """
        return self._normalize_ipw

    @property
    def trimming_rule(self):
        """
        Specifies the used trimming rule.
        """
        return self._trimming_rule

    @property
    def trimming_threshold(self):
        """
        Specifies the used trimming threshold.
        """
        return self._trimming_threshold

    @property
    def _score_element_names(self):
        return ["ind_d", "m_z", "g_du_z0", "g_du_z1", "y", "z", "comp_prob"]

    def _compute_ipw_score(self, theta, d, y, prop, z, comp_prob):
        sign = 2 * self.treatment - 1.0
        weights = sign * (z / prop - (1 - z) / (1 - prop)) / comp_prob
        u = (d == self._treatment) * (y <= theta)
        v = -1.0 * self.quantile
        score = weights * u + v
        return score

    def _compute_score(self, psi_elements, coef, inds=None):
        sign = 2 * self.treatment - 1.0
        ind_d = psi_elements["ind_d"]
        m_z = psi_elements["m_z"]
        g_du_z0 = psi_elements["g_du_z0"]
        g_du_z1 = psi_elements["g_du_z1"]
        y = psi_elements["y"]
        z = psi_elements["z"]
        comp_prob = psi_elements["comp_prob"]

        if inds is not None:
            ind_d = psi_elements["ind_d"][inds]
            m_z = psi_elements["m_z"]
            g_du_z0 = psi_elements["g_du_z0"][inds]
            g_du_z1 = psi_elements["g_du_z1"][inds]
            y = psi_elements["y"][inds]
            z = psi_elements["z"][inds]

        score1 = g_du_z1 - g_du_z0
        score2 = (z / m_z) * (ind_d * (y <= coef) - g_du_z1)
        score3 = (1 - z) / (1 - m_z) * (ind_d * (y <= coef) - g_du_z0)
        score = sign * (score1 + score2 - score3) / comp_prob - self.quantile
        return score

    def _compute_score_deriv(self, psi_elements, coef, inds=None):
        sign = 2 * self.treatment - 1.0
        ind_d = psi_elements["ind_d"]
        y = psi_elements["y"]
        m_z = psi_elements["m_z"]
        z = psi_elements["z"]
        comp_prob = psi_elements["comp_prob"]

        if inds is not None:
            ind_d = psi_elements["ind_d"][inds]
            y = psi_elements["y"][inds]
            m_z = psi_elements["m_z"][inds]
            z = psi_elements["z"][inds]

        score_weights = sign * ((z / m_z) - (1 - z) / (1 - m_z)) * ind_d / comp_prob
        u = (y - coef).reshape(-1, 1)
        deriv = self.kde(u, score_weights)

        return deriv

    def _initialize_ml_nuisance_params(self):
        self._params = {
            learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols}
            for learner in ["ml_m_z", "ml_g_du_z0", "ml_g_du_z1", "ml_m_d_z0", "ml_m_d_z1"]
        }

    def _nuisance_est(self, smpls, n_jobs_cv, external_predictions, return_models=False):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y, force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d, force_all_finite=False)
        x, z = check_X_y(x, np.ravel(self._dml_data.z), force_all_finite=False)

        m_z = external_predictions["ml_m_z"] is not None
        m_d_d0 = external_predictions["ml_m_d_z0"] is not None
        m_d_d1 = external_predictions["ml_m_d_z1"] is not None
        g_du_z0 = external_predictions["ml_g_du_z0"] is not None
        g_du_z1 = external_predictions["ml_g_du_z1"] is not None
        ext_preds = [m_z, m_d_d0, m_d_d1, g_du_z0, g_du_z1]

        # create strata for splitting
        strata = self._dml_data.d.reshape(-1, 1) + 2 * self._dml_data.z.reshape(-1, 1)

        # initialize nuisance predictions, targets and models
        if not all(ext_preds):
            m_z_hat = {
                "models": None,
                "targets": np.full(shape=self._dml_data.n_obs, fill_value=np.nan),
                "preds": np.full(shape=self._dml_data.n_obs, fill_value=np.nan),
            }
            m_d_z0_hat = {
                "models": None,
                "targets": np.full(shape=self._dml_data.n_obs, fill_value=np.nan),
                "preds": np.full(shape=self._dml_data.n_obs, fill_value=np.nan),
            }
            m_d_z1_hat = {
                "models": None,
                "targets": np.full(shape=self._dml_data.n_obs, fill_value=np.nan),
                "preds": np.full(shape=self._dml_data.n_obs, fill_value=np.nan),
            }
            g_du_z0_hat = {
                "models": None,
                "targets": np.full(shape=self._dml_data.n_obs, fill_value=np.nan),
                "preds": np.full(shape=self._dml_data.n_obs, fill_value=np.nan),
            }
            g_du_z1_hat = {
                "models": None,
                "targets": np.full(shape=self._dml_data.n_obs, fill_value=np.nan),
                "preds": np.full(shape=self._dml_data.n_obs, fill_value=np.nan),
            }

            # initialize models
            fitted_models = {}
            for learner in self.params_names:
                # set nuisance model parameters
                est_params = self._get_params(learner)
                if est_params is not None:
                    fitted_models[learner] = [
                        clone(self._learner[learner]).set_params(**est_params[i_fold]) for i_fold in range(self.n_folds)
                    ]
                else:
                    fitted_models[learner] = [clone(self._learner[learner]) for i_fold in range(self.n_folds)]
            ipw_vec = np.full(shape=self.n_folds, fill_value=np.nan)
        elif any(ext_preds) and not any(ext_preds):
            raise ValueError("External predictions for all estimations or for none are required.")
        else:
            m_z_hat = {
                "models": None,
                "targets": np.full(shape=self._dml_data.n_obs, fill_value=np.nan),
                "preds": external_predictions["ml_m_z"],
            }
            m_d_z0_hat = {
                "models": None,
                "targets": np.full(shape=self._dml_data.n_obs, fill_value=np.nan),
                "preds": external_predictions["ml_m_d_z0"],
            }
            m_d_z1_hat = {
                "models": None,
                "targets": np.full(shape=self._dml_data.n_obs, fill_value=np.nan),
                "preds": external_predictions["ml_m_d_z1"],
            }
            g_du_z0_hat = {
                "models": None,
                "targets": np.full(shape=self._dml_data.n_obs, fill_value=np.nan),
                "preds": external_predictions["ml_g_du_z0"],
            }
            g_du_z1_hat = {
                "models": None,
                "targets": np.full(shape=self._dml_data.n_obs, fill_value=np.nan),
                "preds": external_predictions["ml_g_du_z1"],
            }

        # calculate nuisance functions over different folds
        if not all(ext_preds):
            for i_fold in range(self.n_folds):
                train_inds = smpls[i_fold][0]
                test_inds = smpls[i_fold][1]

                # start nested crossfitting
                train_inds_1, train_inds_2 = train_test_split(
                    train_inds, test_size=0.5, random_state=42, stratify=strata[train_inds]
                )
                smpls_prelim = [
                    (train, test)
                    for train, test in StratifiedKFold(n_splits=self.n_folds).split(X=train_inds_1, y=strata[train_inds_1])
                ]

                d_train_1 = d[train_inds_1]
                y_train_1 = y[train_inds_1]
                x_train_1 = x[train_inds_1, :]
                z_train_1 = z[train_inds_1]

                # preliminary propensity for z
                ml_m_z_prelim = clone(fitted_models["ml_m_z"][i_fold])
                m_z_hat_prelim = _dml_cv_predict(ml_m_z_prelim, x_train_1, z_train_1,
                                                 method="predict_proba", smpls=smpls_prelim)[
                    "preds"
                ]

                m_z_hat_prelim = _trimm(m_z_hat_prelim, self.trimming_rule, self.trimming_threshold)
                if self._normalize_ipw:
                    m_z_hat_prelim = _normalize_ipw(m_z_hat_prelim, z_train_1)

                # propensity for d == 1 cond. on z == 0 (training set 1)
                z0_train_1 = z_train_1 == 0
                x_z0_train_1 = x_train_1[z0_train_1, :]
                d_z0_train_1 = d_train_1[z0_train_1]
                ml_m_d_z0_prelim = clone(fitted_models["ml_m_d_z0"][i_fold])
                ml_m_d_z0_prelim.fit(x_z0_train_1, d_z0_train_1)
                m_d_z0_hat_prelim = _predict_zero_one_propensity(ml_m_d_z0_prelim, x_train_1)

                # propensity for d == 1 cond. on z == 1 (training set 1)
                z1_train_1 = z_train_1 == 1
                x_z1_train_1 = x_train_1[z1_train_1, :]
                d_z1_train_1 = d_train_1[z1_train_1]
                ml_m_d_z1_prelim = clone(fitted_models["ml_m_d_z1"][i_fold])
                ml_m_d_z1_prelim.fit(x_z1_train_1, d_z1_train_1)
                m_d_z1_hat_prelim = _predict_zero_one_propensity(ml_m_d_z1_prelim, x_train_1)

                # preliminary estimate of theta_2_aux
                comp_prob_prelim = np.mean(
                    m_d_z1_hat_prelim
                    - m_d_z0_hat_prelim
                    + z_train_1 / m_z_hat_prelim * (d_train_1 - m_d_z1_hat_prelim)
                    - (1 - z_train_1) / (1 - m_z_hat_prelim) * (d_train_1 - m_d_z0_hat_prelim)
                )

                # preliminary ipw estimate
                def ipw_score(theta):
                    res = np.mean(
                        self._compute_ipw_score(theta, d_train_1, y_train_1, m_z_hat_prelim, z_train_1, comp_prob_prelim)
                    )
                    return res

                _, bracket_guess = _get_bracket_guess(ipw_score, self._coef_start_val, self._coef_bounds)
                ipw_est = _solve_ipw_score(ipw_score=ipw_score, bracket_guess=bracket_guess)
                ipw_vec[i_fold] = ipw_est

                # use the preliminary estimates to fit the nuisance parameters on train_2
                d_train_2 = d[train_inds_2]
                y_train_2 = y[train_inds_2]
                x_train_2 = x[train_inds_2, :]
                z_train_2 = z[train_inds_2]

                # define test observations
                d_test = d[test_inds]
                y_test = y[test_inds]
                x_test = x[test_inds, :]
                z_test = z[test_inds]

                # propensity for (D == treatment)*Ind(Y <= ipq_est) cond. on z == 0
                z0_train_2 = z_train_2 == 0
                x_z0_train_2 = x_train_2[z0_train_2, :]
                du_z0_train_2 = (d_train_2[z0_train_2] == self._treatment) * (y_train_2[z0_train_2] <= ipw_est)
                fitted_models["ml_g_du_z0"][i_fold].fit(x_z0_train_2, du_z0_train_2)
                g_du_z0_hat["preds"][test_inds] = _predict_zero_one_propensity(fitted_models["ml_g_du_z0"][i_fold], x_test)

                # propensity for (D == treatment)*Ind(Y <= ipq_est) cond. on z == 1
                z1_train_2 = z_train_2 == 1
                x_z1_train_2 = x_train_2[z1_train_2, :]
                du_z1_train_2 = (d_train_2[z1_train_2] == self._treatment) * (y_train_2[z1_train_2] <= ipw_est)
                fitted_models["ml_g_du_z1"][i_fold].fit(x_z1_train_2, du_z1_train_2)
                g_du_z1_hat["preds"][test_inds] = _predict_zero_one_propensity(fitted_models["ml_g_du_z1"][i_fold], x_test)

                # the predictions of both should only be evaluated conditional on z == 0 or z == 1
                test_inds_z0 = test_inds[z_test == 0]
                test_inds_z1 = test_inds[z_test == 1]
                g_du_z0_hat["targets"][test_inds_z0] = (
                    1.0 * (d_test[z_test == 0] == self._treatment) * (y_test[z_test == 0] <= ipw_est)
                )
                g_du_z1_hat["targets"][test_inds_z1] = (
                    1.0 * (d_test[z_test == 1] == self._treatment) * (y_test[z_test == 1] <= ipw_est)
                )

                # refit nuisance elements for the local potential quantile
                z_train = z[train_inds]
                x_train = x[train_inds]
                d_train = d[train_inds]

                # refit propensity for z (whole training set)
                fitted_models["ml_m_z"][i_fold].fit(x_train, z_train)
                m_z_hat["preds"][test_inds] = _predict_zero_one_propensity(fitted_models["ml_m_z"][i_fold], x_test)

                # refit propensity for d == 1 cond. on z == 0 (whole training set)
                z0_train = z_train == 0
                x_z0_train = x_train[z0_train, :]
                d_z0_train = d_train[z0_train]
                fitted_models["ml_m_d_z0"][i_fold].fit(x_z0_train, d_z0_train)
                m_d_z0_hat["preds"][test_inds] = _predict_zero_one_propensity(fitted_models["ml_m_d_z0"][i_fold], x_test)

                # propensity for d == 1 cond. on z == 1 (whole training set)
                x_z1_train = x_train[z_train == 1, :]
                d_z1_train = d_train[z_train == 1]
                fitted_models["ml_m_d_z1"][i_fold].fit(x_z1_train, d_z1_train)
                m_d_z1_hat["preds"][test_inds] = _predict_zero_one_propensity(fitted_models["ml_m_d_z1"][i_fold], x_test)

        # save targets and models
        m_z_hat["targets"] = z

        # set targets to relevant subsample
        g_du_z0_hat["targets"] = _cond_targets(g_du_z0_hat["targets"], cond_sample=(z == 0))
        g_du_z1_hat["targets"] = _cond_targets(g_du_z1_hat["targets"], cond_sample=(z == 1))

        # the predictions of both should only be evaluated conditional on z == 0 or z == 1
        m_d_z0_hat["targets"] = _cond_targets(d, cond_sample=(z == 0))
        m_d_z1_hat["targets"] = _cond_targets(d, cond_sample=(z == 1))

        if return_models:
            m_z_hat["models"] = fitted_models["ml_m_z"]
            m_d_z0_hat["models"] = fitted_models["ml_m_d_z0"]
            m_d_z1_hat["models"] = fitted_models["ml_m_d_z1"]
            g_du_z0_hat["models"] = fitted_models["ml_g_du_z0"]
            g_du_z1_hat["models"] = fitted_models["ml_g_du_z1"]

        # clip propensities
        m_z_hat_adj = _trimm(m_z_hat["preds"], self.trimming_rule, self.trimming_threshold)

        if self._normalize_ipw:
            m_z_hat_adj = _normalize_ipw(m_z_hat_adj, z)

        # this could be adjusted to be compatible with dml1
        # estimate final nuisance parameter
        comp_prob_hat = np.mean(
            m_d_z1_hat["preds"] - m_d_z0_hat["preds"]
            + z / m_z_hat_adj * (d - m_d_z1_hat["preds"])
            - (1 - z) / (1 - m_z_hat_adj) * (d - m_d_z0_hat["preds"])
        )

        if not all(ext_preds):
            # readjust start value for minimization
            self._coef_start_val = np.mean(ipw_vec)

        psi_elements = {
            "ind_d": d == self._treatment,
            "m_z": m_z_hat_adj,
            "g_du_z0": g_du_z0_hat["preds"],
            "g_du_z1": g_du_z1_hat["preds"],
            "y": y,
            "z": z,
            "comp_prob": comp_prob_hat,
        }
        preds = {
            "predictions": {
                "ml_m_z": m_z_hat["preds"],
                "ml_m_d_z0": m_d_z0_hat["preds"],
                "ml_m_d_z1": m_d_z1_hat["preds"],
                "ml_g_du_z0": g_du_z0_hat["preds"],
                "ml_g_du_z1": g_du_z1_hat["preds"],
            },
            "targets": {
                "ml_m_z": m_z_hat["targets"],
                "ml_m_d_z0": m_d_z0_hat["targets"],
                "ml_m_d_z1": m_d_z1_hat["targets"],
                "ml_g_du_z0": g_du_z0_hat["targets"],
                "ml_g_du_z1": g_du_z1_hat["targets"],
            },
            "models": {
                "ml_m_z": m_z_hat["models"],
                "ml_m_d_z0": m_d_z0_hat["models"],
                "ml_m_d_z1": m_d_z1_hat["models"],
                "ml_g_du_z0": g_du_z0_hat["models"],
                "ml_g_du_z1": g_du_z1_hat["models"],
            },
        }
        return psi_elements, preds

    def _nuisance_tuning(
        self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search
    ):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y, force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d, force_all_finite=False)
        x, z = check_X_y(x, np.ravel(self._dml_data.z), force_all_finite=False)

        if scoring_methods is None:
            scoring_methods = {"ml_m_z": None, "ml_m_d_z0": None, "ml_m_d_z1": None, "ml_g_du_z0": None, "ml_g_du_z1": None}

        train_inds = [train_index for (train_index, _) in smpls]
        train_inds_z0 = [np.intersect1d(np.where(z == 0)[0], train) for train, _ in smpls]
        train_inds_z1 = [np.intersect1d(np.where(z == 1)[0], train) for train, _ in smpls]
        # use a very crude approximation of ipw_est
        approx_quant = np.quantile(y[d == self.treatment], self.quantile)
        du = (d == self.treatment) * (y <= approx_quant)

        m_z_tune_res = _dml_tune(
            z,
            x,
            train_inds,
            self._learner["ml_m_z"],
            param_grids["ml_m_z"],
            scoring_methods["ml_m_z"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )
        m_d_z0_tune_res = _dml_tune(
            d,
            x,
            train_inds_z0,
            self._learner["ml_m_d_z0"],
            param_grids["ml_m_d_z0"],
            scoring_methods["ml_m_d_z0"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )
        m_d_z1_tune_res = _dml_tune(
            d,
            x,
            train_inds_z1,
            self._learner["ml_m_d_z1"],
            param_grids["ml_m_d_z1"],
            scoring_methods["ml_m_d_z1"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )
        g_du_z0_tune_res = _dml_tune(
            du,
            x,
            train_inds_z0,
            self._learner["ml_g_du_z0"],
            param_grids["ml_g_du_z0"],
            scoring_methods["ml_g_du_z0"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )
        g_du_z1_tune_res = _dml_tune(
            du,
            x,
            train_inds_z1,
            self._learner["ml_g_du_z1"],
            param_grids["ml_g_du_z1"],
            scoring_methods["ml_g_du_z1"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )

        m_z_best_params = [xx.best_params_ for xx in m_z_tune_res]
        m_d_z0_best_params = [xx.best_params_ for xx in m_d_z0_tune_res]
        m_d_z1_best_params = [xx.best_params_ for xx in m_d_z1_tune_res]
        g_du_z0_best_params = [xx.best_params_ for xx in g_du_z0_tune_res]
        g_du_z1_best_params = [xx.best_params_ for xx in g_du_z1_tune_res]

        params = {
            "ml_m_z": m_z_best_params,
            "ml_m_d_z0": m_d_z0_best_params,
            "ml_m_d_z1": m_d_z1_best_params,
            "ml_g_du_z0": g_du_z0_best_params,
            "ml_g_du_z1": g_du_z1_best_params,
        }
        tune_res = {
            "ml_m_z": m_z_tune_res,
            "ml_m_d_z0": m_d_z0_tune_res,
            "ml_m_d_z1": m_d_z1_tune_res,
            "ml_g_du_z0": g_du_z0_tune_res,
            "ml_g_du_z1": g_du_z1_tune_res,
        }

        res = {"params": params, "tune_res": tune_res}

        return res

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError(
                "The data must be of DoubleMLData type. " f"{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed."
            )
        _check_zero_one_treatment(self)
        one_instr = obj_dml_data.n_instr == 1
        err_msg = (
            "Incompatible data. "
            "To fit an LPQ model with DML "
            "exactly one binary variable with values 0 and 1 "
            "needs to be specified as instrumental variable."
        )
        if one_instr:
            binary_instr = type_of_target(obj_dml_data.z) == "binary"
            zero_one_instr = np.all((np.power(obj_dml_data.z, 2) - obj_dml_data.z) == 0)
            if not (one_instr & binary_instr & zero_one_instr):
                raise ValueError(err_msg)
        else:
            raise ValueError(err_msg)
        return

    def _sensitivity_element_est(self, preds):
        pass
