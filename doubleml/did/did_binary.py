import warnings

import numpy as np
from sklearn.utils import check_X_y

from doubleml.data.panel_data import DoubleMLPanelData
from doubleml.did.utils._did_utils import (
    _check_anticipation_periods,
    _check_control_group,
    _check_gt_combination,
    _check_gt_values,
    _get_id_positions,
    _get_never_treated_value,
    _is_never_treated,
    _set_id_positions,
)
from doubleml.double_ml import DoubleML
from doubleml.double_ml_score_mixins import LinearScoreMixin
from doubleml.utils._checks import (
    _check_bool,
    _check_finite_predictions,
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
    obj_dml_data : :class:`DoubleMLPanelData` object
        The :class:`DoubleMLPanelData` object providing the data and specifying the variables for the causal model.

    g_value : int
        The value indicating the treatment group (first period with treatment).
        Default is ``None``. This implements the case for the smallest, non-zero value of G.

    t_value_pre : int
        The value indicating the baseline pre-treatment period.

    t_value_eval : int
        The value indicating the period for evaluation.

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

    control_group : str
        Specifies the control group. Either ``'never_treated'`` or ``'not_yet_treated'``.
        Default is ``'never_treated'``.

    anticipation_periods : int
        Number of anticipation periods. Default is ``0``.

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

    print_periods : bool
        Indicates whether to print information about the evaluated periods.
        Default is ``False``.

    """

    def __init__(
        self,
        obj_dml_data,
        g_value,
        t_value_pre,
        t_value_eval,
        ml_g,
        ml_m=None,
        control_group="never_treated",
        anticipation_periods=0,
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
        g_values = self._dml_data.g_values
        t_values = self._dml_data.t_values

        _check_bool(print_periods, "print_periods")
        self._print_periods = print_periods

        self._control_group = _check_control_group(control_group)
        self._never_treated_value = _get_never_treated_value(g_values)
        self._anticipation_periods = _check_anticipation_periods(anticipation_periods)

        _check_gt_combination(
            (g_value, t_value_pre, t_value_eval), g_values, t_values, self.never_treated_value, self.anticipation_periods
        )
        self._g_value = g_value
        self._t_value_pre = t_value_pre
        self._t_value_eval = t_value_eval

        # check if post_treatment evaluation
        if g_value <= t_value_eval:
            post_treatment = True
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
        id_original = self._dml_data.id_var_unique
        if not np.all(np.isin(id_panel_data, id_original)):
            raise ValueError("The id values in the panel data are not a subset of the original id values.")

        # Find position of id_panel_data in original data
        # These entries should be replaced by nuisance predictions, all others should be set to 0.
        self._id_positions = np.searchsorted(id_original, id_panel_data)

        # Numeric values for positions of the entries in id_panel_data inside id_original
        # np.nonzero(np.isin(id_original, id_panel_data))
        self._n_subset = self._panel_data_wide.shape[0]
        self._n_obs = self._n_subset  # Effective sample size used for resampling
        self._n_treated_subset = self._panel_data_wide["G_indicator"].sum()

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
            self.draw_sample_splitting()

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

    def __str__(self):
        class_name = self.__class__.__name__
        header = f"================== {class_name} Object ==================\n"
        data_summary = self._dml_data._data_summary_str()
        score_info = (
            f"Score function: {str(self.score)}\n"
            f"Treatment group: {str(self.g_value)}\n"
            f"Pre-treatment period: {str(self.t_value_pre)}\n"
            f"Evaluation period: {str(self.t_value_eval)}\n"
            f"Control group: {str(self.control_group)}\n"
            f"Anticipation periods: {str(self.anticipation_periods)}\n"
            f"Effective sample size: {str(self.n_obs)}\n"
        )
        learner_info = ""
        for key, value in self.learner.items():
            learner_info += f"Learner {key}: {str(value)}\n"
        if self.nuisance_loss is not None:
            learner_info += "Out-of-sample Performance:\n"
            is_classifier = [value for value in self._is_classifier.values()]
            is_regressor = [not value for value in is_classifier]
            if any(is_regressor):
                learner_info += "Regression:\n"
                for learner in [key for key, value in self._is_classifier.items() if value is False]:
                    learner_info += f"Learner {learner} RMSE: {self.nuisance_loss[learner]}\n"
            if any(is_classifier):
                learner_info += "Classification:\n"
                for learner in [key for key, value in self._is_classifier.items() if value is True]:
                    learner_info += f"Learner {learner} Log Loss: {self.nuisance_loss[learner]}\n"

        if self._is_cluster_data:
            resampling_info = (
                f"No. folds per cluster: {self._n_folds_per_cluster}\n"
                f"No. folds: {self.n_folds}\n"
                f"No. repeated sample splits: {self.n_rep}\n"
            )
        else:
            resampling_info = f"No. folds: {self.n_folds}\nNo. repeated sample splits: {self.n_rep}\n"
        fit_summary = str(self.summary)
        res = (
            header
            + "\n------------------ Data summary      ------------------\n"
            + data_summary
            + "\n------------------ Score & algorithm ------------------\n"
            + score_info
            + "\n------------------ Machine learner   ------------------\n"
            + learner_info
            + "\n------------------ Resampling        ------------------\n"
            + resampling_info
            + "\n------------------ Fit summary       ------------------\n"
            + fit_summary
        )
        return res

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
    def never_treated_value(self):
        """
        The value indicating that a unit was never treated.
        """
        return self._never_treated_value

    @property
    def post_treatment(self):
        """
        Indicates whether the evaluation period is after the treatment period.
        """
        return self._post_treatment

    @property
    def control_group(self):
        """
        The control group.
        """
        return self._control_group

    @property
    def anticipation_periods(self):
        """
        The number of anticipation periods.
        """
        return self._anticipation_periods

    @property
    def panel_data_wide(self):
        """
        The preprocessed panel data in wide format.
        """
        return self._panel_data_wide

    @property
    def id_positions(self):
        """
        The positions of the id values in the original data.
        """
        return self._id_positions

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

    @property
    def n_obs(self):
        """
        The number of observations used for estimation.
        """
        return self._n_subset

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

        one_treat = obj_dml_data.n_treat == 1
        if not (one_treat):
            raise ValueError(
                "Incompatible data. "
                "To fit an DID model with DML "
                "exactly one variable needs to be specified as treatment variable."
            )
        _check_gt_values(obj_dml_data.g_values, obj_dml_data.t_values)
        return

    def _preprocess_data(self, g_value, pre_t, eval_t):
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
        never_treated = _is_never_treated(data_subset[g_col], self.never_treated_value).reshape(-1)
        if self.control_group == "never_treated":
            C_indicator = never_treated.astype(int)

        elif self.control_group == "not_yet_treated":
            # adjust max_g_value for anticipation periods
            t_values = self._dml_data.t_values
            max_g_value = t_values[min(np.where(t_values == eval_t)[0][0] + self.anticipation_periods, len(t_values) - 1)]
            # not in G just as a additional check
            later_treated = (data_subset[g_col] > max_g_value) & (G_indicator == 0)
            not_yet_treated = never_treated | later_treated
            C_indicator = not_yet_treated.astype(int)

        if np.sum(C_indicator) == 0:
            raise ValueError("No observations in the control group.")

        data_subset = data_subset.assign(C_indicator=C_indicator, G_indicator=G_indicator)
        # reduce to relevant subset
        data_subset = data_subset[(data_subset["G_indicator"] == 1) | (data_subset["C_indicator"] == 1)]
        # check if G and C are disjoint
        assert sum(G_indicator & C_indicator) == 0

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

        # Here: d is a binary treatment indicator
        x, y = check_X_y(self._x_panel, self._y_panel, force_all_finite=False)
        x, d = check_X_y(x, self._g_panel, force_all_finite=False)
        # nuisance g
        # get train indices for d == 0
        smpls_d0, smpls_d1 = _get_cond_smpls(smpls, d)

        # nuisance g for d==0
        if external_predictions["ml_g0"] is not None:
            ml_g0_targets = np.full_like(y, np.nan, dtype="float64")
            ml_g0_targets[d == 0] = y[d == 0]
            ml_g0_pred = _get_id_positions(external_predictions["ml_g0"], self.id_positions)
            g_hat0 = {"preds": ml_g0_pred, "targets": ml_g0_targets, "models": None}
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
            ml_g1_targets = np.full_like(y, np.nan, dtype="float64")
            ml_g1_targets[d == 1] = y[d == 1]
            ml_g1_pred = _get_id_positions(external_predictions["ml_g1"], self.id_positions)
            g_hat1 = {"preds": ml_g1_pred, "targets": ml_g1_targets, "models": None}
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
                ml_m_pred = _get_id_positions(external_predictions["ml_m"], self.id_positions)
                m_hat = {"preds": ml_m_pred, "targets": d, "models": None}
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
        p_hat = np.full_like(d, d.mean(), dtype="float64")
        psi_a, psi_b = self._score_elements(y, d, g_hat0["preds"], g_hat1["preds"], m_hat["preds"], p_hat)

        extend_kwargs = {
            "n_obs": self._dml_data.n_obs,
            "id_positions": self.id_positions,
        }
        psi_elements = {
            "psi_a": _set_id_positions(psi_a, fill_value=0.0, **extend_kwargs),
            "psi_b": _set_id_positions(psi_b, fill_value=0.0, **extend_kwargs),
        }
        preds = {
            "predictions": {
                "ml_g0": _set_id_positions(g_hat0["preds"], fill_value=np.nan, **extend_kwargs),
                "ml_g1": _set_id_positions(g_hat1["preds"], fill_value=np.nan, **extend_kwargs),
                "ml_m": _set_id_positions(m_hat["preds"], fill_value=np.nan, **extend_kwargs),
            },
            "targets": {
                "ml_g0": _set_id_positions(g_hat0["targets"], fill_value=np.nan, **extend_kwargs),
                "ml_g1": _set_id_positions(g_hat1["targets"], fill_value=np.nan, **extend_kwargs),
                "ml_m": _set_id_positions(m_hat["targets"], fill_value=np.nan, **extend_kwargs),
            },
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

        return psi_a, psi_b

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

        m_hat = _get_id_positions(preds["predictions"]["ml_m"], self.id_positions)
        g_hat0 = _get_id_positions(preds["predictions"]["ml_g0"], self.id_positions)
        g_hat1 = _get_id_positions(preds["predictions"]["ml_g1"], self.id_positions)

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

        extend_kwargs = {
            "n_obs": self._dml_data.n_obs,
            "id_positions": self.id_positions,
            "fill_value": 0.0,
        }

        # add scaling to make variance estimation consistent (sample size difference)
        scaling = self._dml_data.n_obs / self._n_subset
        element_dict = {
            "sigma2": sigma2,
            "nu2": nu2,
            "psi_sigma2": scaling * _set_id_positions(psi_sigma2, **extend_kwargs),
            "psi_nu2": scaling * _set_id_positions(psi_nu2, **extend_kwargs),
            "riesz_rep": scaling * _set_id_positions(rr, **extend_kwargs),
        }
        return element_dict

    def sensitivity_benchmark(self, benchmarking_set, fit_args=None):
        """
        Computes a benchmark for a given set of features.
        Returns a DataFrame containing the corresponding values for cf_y, cf_d, rho and the change in estimates.

        Parameters
        ----------
        benchmarking_set : list
            List of features to be used for benchmarking.

        fit_args : dict, optional
            Additional arguments for the fit method.
            Default is None.

        Returns
        -------
        benchmark_results : pandas.DataFrame
            Benchmark results.
        """
        if self.score == "experimental":
            warnings.warn(
                "Sensitivity benchmarking for experimental score may not be meaningful. "
                "Consider using score='observational' for conditional treatment assignment.",
                UserWarning,
            )

        return super().sensitivity_benchmark(benchmarking_set, fit_args)
