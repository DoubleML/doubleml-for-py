import warnings

import numpy as np
from sklearn.utils import check_X_y

from doubleml.data.panel_data import DoubleMLPanelData
from doubleml.double_ml import DoubleML
from doubleml.double_ml_score_mixins import LinearScoreMixin
from doubleml.utils._checks import (
    _check_bool,
    _check_finite_predictions,
    _check_integer,
    _check_is_propensity,
    _check_score,
    _check_trimming,
)
from doubleml.utils._estimation import _dml_cv_predict, _dml_tune, _get_cond_smpls
from doubleml.utils._propensity_score import _trimm


class DoubleMLDIDBinary(LinearScoreMixin, DoubleML):
    """Double machine learning for difference-in-differences models with panel data (binary setting in terms of group and time
     combinations).

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.
        The data input has to be in a panel format with one row per time period per individual.

    ml_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`g_0(d,X) = E[Y_1-Y_0|D=d, X]`.
        For a binary outcome variable :math:`Y` (with values 0 and 1), a classifier implementing ``fit()`` and
        ``predict_proba()`` can also be specified. If :py:func:`sklearn.base.is_classifier` returns ``True``,
        ``predict_proba()`` is used otherwise ``predict()``.

    ml_m : classifier implementing ``fit()`` and ``predict_proba()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance function :math:`m_0(X) = E[D=1|X]`.
        Only relevant for ``score='observational'``.

    g_value: int
        The value indicating the treatment group (first period with treatment).
        Default is ``None``. This implements the case for the smallest, non-zero value of G.

    t_value: int
        The value indicating the base period for evaluation.
        Default is ``None``. This implements the case of the smallest value of T.

    pre_period : int
        The value indicating the pre-treatment period.
        Default is ``None`` implementing the period before the treatment period specified by ``g_value``.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitons for the sample splitting.
        Default is ``1``.

    score : str
        A str (``'observational'`` or ``'experimental'``) specifying the score function.
        The ``'experimental'`` scores refers to an A/B setting, where the treatment is independent
        from the pretreatment covariates.
        Default is ``'observational'``.

    in_sample_normalization : bool
        Indicates whether to use a sligthly different normalization from Sant'Anna and Zhao (2020).
        Default is ``True``.

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
    TODO: Add example
    """

    def __init__(
        self,
        obj_dml_data,
        ml_g,
        ml_m=None,
        g_value=None,
        t_value_pre=None,
        t_value_eval=None,
        control_group="never_treated",
        n_folds=5,
        n_rep=1,
        score="observational",
        in_sample_normalization=True,
        trimming_rule="truncate",
        trimming_threshold=1e-2,
        draw_sample_splitting=True,
        print_periods=False,
    ):

        super().__init__(obj_dml_data, n_folds, n_rep, score, draw_sample_splitting=False)

        self._check_data(self._dml_data)
        _check_bool(print_periods, "print_periods")
        self._print_periods = print_periods

        # TODO: Check if we want to keep the g_values & t_values from the backend
        # as they might differ from the ones processed in DoubleMLDIDMULTI
        g_values = self._dml_data.g_values
        t_values = self._dml_data.t_values

        valid_control_groups = ["never_treated", "not_yet_treated"]
        if control_group not in valid_control_groups:
            raise ValueError(f"The control group has to be one of {valid_control_groups}. " + f"{control_group} was passed.")
        self._control_group = control_group

        # TODO: Do we really want to use default values for g and t?
        # Defaults for g and t values
        if g_value is None:
            g_value = int(g_values[g_values > 0].min())
        if t_value_pre is None:
            t_value_pre = int(t_values[t_values < g_value].max())
        if t_value_eval is None:
            t_value_eval = int(t_values[t_values > g_value].min())

        _check_integer(g_value, "g_value", 0)
        _check_integer(t_value_pre, "t_value_pre", 0)
        _check_integer(t_value_eval, "t_value_eval", 0)

        # check if g_value and t_value are in the set of g_values and t_values
        if g_value not in g_values:
            raise ValueError(f"The value {g_value} is not in the set of treatment group values {g_values}.")
        if g_value == 0:
            raise ValueError("The never treated group is not allowed as treatment group (g_value=0).")
        if t_value_pre not in t_values:
            raise ValueError(f"The value {t_value_pre} is not in the set of evaluation period values {t_values}.")
        if t_value_eval not in t_values:
            raise ValueError(f"The value {t_value_eval} is not in the set of evaluation period values {t_values}.")

        self._g_value = g_value
        self._t_value_pre = t_value_pre
        self._t_value_eval = t_value_eval

        # check if post_treatment evaluation
        if g_value <= t_value_eval:
            post_treatment = True
            if t_value_pre <= 0:
                # TODO: What does this mean?
                print(
                    f"No pre-treatment period available for group first treated in {g_value}.\n\
                        Units from this group are dropped."
                )
        else:
            post_treatment = False

        self._post_treatment = post_treatment

        if self._print_periods:
            print(
                f"Evaluation of ATT({g_value}, {t_value_eval}), with pre-treatment period {t_value_pre},\n"
                + f"post-treatment: {post_treatment}. Control group: {control_group}.\n"
            )

        # Preprocess data
        # Y1, Y0 might be needed if we want to support custom estimators and scores; currently only output y_diff
        self._panel_data_wide = self._preprocess_data(self._g_value, self._t_value_pre, self._t_value_eval)

        # Handling id values to match pairwise evaluation & simultaneous inference
        id_panel_data = self._panel_data_wide[self._dml_data.id_col].values

        # Psi: 1 x n_obs (where n_obs = unique(id))
        # original unique id values
        id_original = self._dml_data.id_var_unique

        # Find position of id_panel_data in original data
        # These entries should be replaced by nuisance predictions, all others should be set to 0.
        id_subset = np.where(np.isin(id_original, id_panel_data))
        self._id_subset = id_subset

        # Numeric values for positions of the entries in id_panel_data inside id_original
        # np.nonzero(np.isin(id_original, id_panel_data))
        self._n_subset = self._panel_data_wide.shape[0]
        self._n_treated_subset = self._panel_data_wide["G_indicator"].sum()

        # TODO: Is this necessary?
        # Save x and y for later ML estimation
        self._x_panel = self._panel_data_wide.loc[:, self._dml_data.x_cols].values
        self._y_panel = self._panel_data_wide.loc[:, "y_diff"].values
        self._g_panel = self._panel_data_wide.loc[:, "G_indicator"].values

        valid_scores = ["observational", "experimental"]
        _check_score(self.score, valid_scores, allow_callable=False)

        self._in_sample_normalization = in_sample_normalization
        if not isinstance(self.in_sample_normalization, bool):
            raise TypeError(
                "in_sample_normalization indicator has to be boolean. "
                + f"Object of type {str(type(self.in_sample_normalization))} passed."
            )

        # set stratication for resampling
        self._strata = self._panel_data_wide["G_indicator"]
        if draw_sample_splitting:
            # TODO: Handle n_obs: n_obs_subset is likely smaller than n_obs!
            self.draw_sample_splitting(n_obs=self._n_subset)

        # check learners
        ml_g_is_classifier = self._check_learner(ml_g, "ml_g", regressor=True, classifier=True)
        if self.score == "observational":
            _ = self._check_learner(ml_m, "ml_m", regressor=False, classifier=True)
            self._learner = {"ml_g": ml_g, "ml_m": ml_m}
        else:
            assert self.score == "experimental"
            if ml_m is not None:
                warnings.warn(
                    (
                        'A learner ml_m has been provided for score = "experimental" but will be ignored. '
                        "A learner ml_m is not required for estimation."
                    )
                )
            self._learner = {"ml_g": ml_g}

        if ml_g_is_classifier:
            if obj_dml_data.binary_outcome:
                self._predict_method = {"ml_g": "predict_proba"}
            else:
                raise ValueError(
                    f"The ml_g learner {str(ml_g)} was identified as classifier "
                    "but the outcome variable is not binary with values 0 and 1."
                )
        else:
            self._predict_method = {"ml_g": "predict"}

        if "ml_m" in self._learner:
            self._predict_method["ml_m"] = "predict_proba"
        self._initialize_ml_nuisance_params()

        self._trimming_rule = trimming_rule
        self._trimming_threshold = trimming_threshold
        _check_trimming(self._trimming_rule, self._trimming_threshold)

        self._sensitivity_implemented = True
        self._external_predictions_implemented = True

    @property
    def g_value(self):
        """
        The value indicating the treatment group (first period with treatment).
        """
        return self._g_value

    @property
    def t_value_eval(self):
        """
        The value indicating the evaluation period.
        """
        return self._t_value_eval

    @property
    def t_value_pre(self):
        """
        The value indicating the pre-treatment period.
        """
        return self._t_value_pre

    @property
    def post_treatment(self):
        """
        Indicates whether the evaluation period is after the treatment period.
        """
        return self._post_treatment

    @property
    def panel_data_wide(self):
        """
        The preprocessed panel data in wide format.
        """
        return self._panel_data_wide

    @property
    def in_sample_normalization(self):
        """
        Indicates whether the in sample normalization of weights are used.
        """
        return self._in_sample_normalization

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

    def _initialize_ml_nuisance_params(self):
        if self.score == "observational":
            valid_learner = ["ml_g0", "ml_g1", "ml_m"]
        else:
            assert self.score == "experimental"
            valid_learner = ["ml_g0", "ml_g1"]
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in valid_learner}

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLPanelData):
            raise TypeError(
                "For repeated outcomes the data must be of DoubleMLPanelData type. "
                f"{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed."
            )
        if obj_dml_data.z_cols is not None:
            raise NotImplementedError(
                "Incompatible data. " + " and ".join(obj_dml_data.z_cols) + " have been set as instrumental variable(s). "
                "At the moment there are not DiD models with instruments implemented."
            )

        # TODO: Update checks!
        one_treat = obj_dml_data.n_treat == 1
        if not (one_treat):
            raise ValueError(
                "Incompatible data. "
                "To fit an DID model with DML "
                "exactly one variable needs to be specified as treatment variable."
            )
        return

    def _preprocess_data(self, g_value, pre_t, eval_t):
        # TODO: Check if copy is necessary
        data = self._dml_data.data

        y_col = self._dml_data.y_col
        t_col = self._dml_data.t_col
        id_col = self._dml_data.id_col
        g_col = self._dml_data.g_col

        # relevent data subset
        data_subset_indicator = data[t_col].isin([pre_t, eval_t])
        data_subset = data[data_subset_indicator].sort_values(by=[id_col, t_col])

        # Construct G (treatment group) indicating treatment period in g
        G_indicator = (data_subset[g_col] == g_value).astype(int)

        # Construct C (control group) indicating never treated or not yet treated
        never_treated = data_subset[g_col] == 0
        if self._control_group == "never_treated":
            C_indicator = never_treated.astype(int)

        elif self._control_group == "not_yet_treated":
            # C_indicator = Never treated or treated in period after eval_t (and not in g)
            # TODO: Check again if this_data['G_indicator'] == 0 makes sense
            later_treated = (data_subset[g_col] > max(pre_t, eval_t)) & (data_subset["G_indicator"] == 0)
            not_yet_treated = never_treated | later_treated
            C_indicator = not_yet_treated.astype(int)

        data_subset = data_subset.assign(C_indicator=C_indicator, G_indicator=G_indicator)
        # reduce to relevant subset
        data_subset = data_subset[(data_subset["G_indicator"] == 1) | (data_subset["C_indicator"] == 1)]
        # TODO: Add checks e.g. T != C
        # TODO: Enforce Balanced panel data!

        # Alternatively, use .shift() (check if time ordering is correct)
        # y_diff = this_data.groupby(id_col)[y_col].shift(-1)
        y_diff = (
            data_subset[data_subset[t_col] == eval_t][y_col].values - data_subset[data_subset[t_col] == pre_t][y_col].values
        )

        # keep covariates only observations from the first period
        # Data processing from long to wide format
        select_cols = [id_col, "G_indicator", "C_indicator"] + self._dml_data.x_cols
        first_period = data_subset[t_col].min()
        wide_data = data_subset[select_cols][data_subset[t_col] == first_period]
        wide_data = wide_data.assign(y_diff=y_diff)

        return wide_data

    def _nuisance_est(self, smpls, n_jobs_cv, external_predictions, return_models=False):

        # TODO: Decide whether to do preprocessing here or before?
        # Here: d is a binary treatment indicator
        x, y = check_X_y(self._x_panel, self._y_panel, force_all_finite=False)
        x, d = check_X_y(x, self._g_panel, force_all_finite=False)
        # nuisance g
        # get train indices for d == 0
        smpls_d0, smpls_d1 = _get_cond_smpls(smpls, d)

        # nuisance g for d==0
        if external_predictions["ml_g0"] is not None:
            g_hat0 = {"preds": external_predictions["ml_g0"], "targets": None, "models": None}
        else:
            g_hat0 = _dml_cv_predict(
                self._learner["ml_g"],
                x,
                y,
                smpls=smpls_d0,
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_g0"),
                method=self._predict_method["ml_g"],
                return_models=return_models,
            )

            _check_finite_predictions(g_hat0["preds"], self._learner["ml_g"], "ml_g", smpls)
            # adjust target values to consider only compatible subsamples
            g_hat0["targets"] = g_hat0["targets"].astype(float)
            g_hat0["targets"][d == 1] = np.nan

        # nuisance g for d==1
        if external_predictions["ml_g1"] is not None:
            g_hat1 = {"preds": external_predictions["ml_g1"], "targets": None, "models": None}
        else:
            g_hat1 = _dml_cv_predict(
                self._learner["ml_g"],
                x,
                y,
                smpls=smpls_d1,
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_g1"),
                method=self._predict_method["ml_g"],
                return_models=return_models,
            )

            _check_finite_predictions(g_hat1["preds"], self._learner["ml_g"], "ml_g", smpls)
            # adjust target values to consider only compatible subsamples
            g_hat1["targets"] = g_hat1["targets"].astype(float)
            g_hat1["targets"][d == 0] = np.nan

        # only relevant for observational setting
        m_hat = {"preds": None, "targets": None, "models": None}
        if self.score == "observational":
            # nuisance m
            if external_predictions["ml_m"] is not None:
                m_hat = {"preds": external_predictions["ml_m"], "targets": None, "models": None}
            else:
                m_hat = _dml_cv_predict(
                    self._learner["ml_m"],
                    x,
                    d,
                    smpls=smpls,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_m"),
                    method=self._predict_method["ml_m"],
                    return_models=return_models,
                )
            _check_finite_predictions(m_hat["preds"], self._learner["ml_m"], "ml_m", smpls)
            _check_is_propensity(m_hat["preds"], self._learner["ml_m"], "ml_m", smpls, eps=1e-12)
            m_hat["preds"] = _trimm(m_hat["preds"], self.trimming_rule, self.trimming_threshold)

        # nuisance estimates of the uncond. treatment prob.
        p_hat = np.full_like(d, np.nan, dtype="float64")
        for train_index, test_index in smpls:
            p_hat[test_index] = np.mean(d[train_index])

        psi_a, psi_b = self._score_elements(y, d, g_hat0["preds"], g_hat1["preds"], m_hat["preds"], p_hat)

        psi_elements = {"psi_a": psi_a, "psi_b": psi_b}
        preds = {
            "predictions": {"ml_g0": g_hat0["preds"], "ml_g1": g_hat1["preds"], "ml_m": m_hat["preds"]},
            "targets": {"ml_g0": g_hat0["targets"], "ml_g1": g_hat1["targets"], "ml_m": m_hat["targets"]},
            "models": {"ml_g0": g_hat0["models"], "ml_g1": g_hat1["models"], "ml_m": m_hat["models"]},
        }

        return psi_elements, preds

    def _score_elements(self, y, d, g_hat0, g_hat1, m_hat, p_hat):
        # calc residuals
        resid_d0 = y - g_hat0

        if self.score == "observational":
            if self.in_sample_normalization:
                weight_psi_a = np.divide(d, np.mean(d))
                propensity_weight = np.multiply(1.0 - d, np.divide(m_hat, 1.0 - m_hat))
                weight_resid_d0 = np.divide(d, np.mean(d)) - np.divide(propensity_weight, np.mean(propensity_weight))
            else:
                weight_psi_a = np.divide(d, p_hat)
                weight_resid_d0 = np.divide(d - m_hat, np.multiply(p_hat, 1.0 - m_hat))

            psi_b_1 = np.zeros_like(y)

        else:
            assert self.score == "experimental"
            if self.in_sample_normalization:
                weight_psi_a = np.ones_like(y)
                weight_g0 = np.divide(d, np.mean(d)) - 1.0
                weight_g1 = 1.0 - np.divide(d, np.mean(d))
                weight_resid_d0 = np.divide(d, np.mean(d)) - np.divide(1.0 - d, np.mean(1.0 - d))
            else:
                weight_psi_a = np.ones_like(y)
                weight_g0 = np.divide(d, p_hat) - 1.0
                weight_g1 = 1.0 - np.divide(d, p_hat)
                weight_resid_d0 = np.divide(d - p_hat, np.multiply(p_hat, 1.0 - p_hat))

            psi_b_1 = np.multiply(weight_g0, g_hat0) + np.multiply(weight_g1, g_hat1)

        # set score elements
        psi_a = -1.0 * weight_psi_a
        psi_b = psi_b_1 + np.multiply(weight_resid_d0, resid_d0)

        # TODO: Check rescaling by factor n_obs/n_subset
        # scaling_factor_did = self._dml_data.n_obs / self._n_subset
        # psi_b = psi_b * scaling_factor_did
        return psi_a, psi_b

    def _initialize_predictions_and_targets(self):
        # Here: Initialize predictions and targets based on panel data in wide format (n_subset)
        self._predictions = {
            learner: np.full((self._n_subset, self.n_rep, self._dml_data.n_coefs), np.nan) for learner in self.params_names
        }
        self._nuisance_targets = {
            learner: np.full((self._n_subset, self.n_rep, self._dml_data.n_coefs), np.nan) for learner in self.params_names
        }

    def _set_score_elements(self, psi_elements, i_rep, i_treat):
        # Specific implementation for DoubleMLDIDBINARY to account for long vs. wide data format
        if not isinstance(psi_elements, dict):
            raise TypeError(
                "_ml_nuisance_and_score_elements must return score elements in a dict. " f"Got type {str(type(psi_elements))}."
            )
        if not (set(self._score_element_names) == set(psi_elements.keys())):
            raise ValueError(
                "_ml_nuisance_and_score_elements returned incomplete score elements. "
                "Expected dict with keys: " + " and ".join(set(self._score_element_names)) + "."
                "Got dict with keys: " + " and ".join(set(psi_elements.keys())) + "."
            )
        for key in self._score_element_names:
            # set values for psi_a and psi_b entries for i's not in gt subgroup to 0
            self.psi_elements[key][:, i_rep, i_treat] = 0
            self.psi_elements[key][self._id_subset, i_rep, i_treat] = psi_elements[key]

        return

    def _nuisance_tuning(
        self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search
    ):
        x, y = check_X_y(self._x_panel, self._y_panel, force_all_finite=False)
        x, d = check_X_y(x, self._g_panel, force_all_finite=False)

        # get train indices for d == 0 and d == 1
        smpls_d0, smpls_d1 = _get_cond_smpls(smpls, d)

        if scoring_methods is None:
            scoring_methods = {"ml_g": None, "ml_m": None}

        train_inds = [train_index for (train_index, _) in smpls]
        train_inds_d0 = [train_index for (train_index, _) in smpls_d0]
        train_inds_d1 = [train_index for (train_index, _) in smpls_d1]
        g0_tune_res = _dml_tune(
            y,
            x,
            train_inds_d0,
            self._learner["ml_g"],
            param_grids["ml_g"],
            scoring_methods["ml_g"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )
        g1_tune_res = _dml_tune(
            y,
            x,
            train_inds_d1,
            self._learner["ml_g"],
            param_grids["ml_g"],
            scoring_methods["ml_g"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )

        g0_best_params = [xx.best_params_ for xx in g0_tune_res]
        g1_best_params = [xx.best_params_ for xx in g1_tune_res]

        m_tune_res = list()
        if self.score == "observational":
            m_tune_res = _dml_tune(
                d,
                x,
                train_inds,
                self._learner["ml_m"],
                param_grids["ml_m"],
                scoring_methods["ml_m"],
                n_folds_tune,
                n_jobs_cv,
                search_mode,
                n_iter_randomized_search,
            )
            m_best_params = [xx.best_params_ for xx in m_tune_res]
            params = {"ml_g0": g0_best_params, "ml_g1": g1_best_params, "ml_m": m_best_params}
            tune_res = {"g0_tune": g0_tune_res, "g1_tune": g1_tune_res, "m_tune": m_tune_res}
        else:
            params = {"ml_g0": g0_best_params, "ml_g1": g1_best_params}
            tune_res = {"g0_tune": g0_tune_res, "g1_tune": g1_tune_res}

        res = {"params": params, "tune_res": tune_res}

        return res

    def _sensitivity_element_est(self, preds):
        y = self._y_panel
        d = self._g_panel

        m_hat = preds["predictions"]["ml_m"]
        g_hat0 = preds["predictions"]["ml_g0"]
        g_hat1 = preds["predictions"]["ml_g1"]

        g_hat = np.multiply(d, g_hat1) + np.multiply(1.0 - d, g_hat0)
        sigma2_score_element = np.square(y - g_hat)
        sigma2 = np.mean(sigma2_score_element)
        psi_sigma2 = sigma2_score_element - sigma2

        # calc m(W,alpha) and Riesz representer
        p_hat = np.mean(d)
        if self.score == "observational":
            propensity_weight_d0 = np.divide(m_hat, 1.0 - m_hat)
            if self.in_sample_normalization:
                weight_d0 = np.multiply(1.0 - d, propensity_weight_d0)
                mean_weight_d0 = np.mean(weight_d0)

                m_alpha = np.multiply(
                    np.divide(d, p_hat), np.divide(1.0, p_hat) + np.divide(propensity_weight_d0, mean_weight_d0)
                )
                rr = np.divide(d, p_hat) - np.divide(weight_d0, mean_weight_d0)
            else:
                m_alpha = np.multiply(np.divide(d, np.square(p_hat)), (1.0 + propensity_weight_d0))
                rr = np.divide(d, p_hat) - np.multiply(np.divide(1.0 - d, p_hat), propensity_weight_d0)
        else:
            assert self.score == "experimental"
            # the same with or without self-normalization
            m_alpha = np.divide(1.0, p_hat) + np.divide(1.0, 1.0 - p_hat)
            rr = np.divide(d, p_hat) - np.divide(1.0 - d, 1.0 - p_hat)

        nu2_score_element = np.multiply(2.0, m_alpha) - np.square(rr)
        nu2 = np.mean(nu2_score_element)
        psi_nu2 = nu2_score_element - nu2

        element_dict = {
            "sigma2": sigma2,
            "nu2": nu2,
            "psi_sigma2": psi_sigma2,
            "psi_nu2": psi_nu2,
            "riesz_rep": rr,
        }
        return element_dict
