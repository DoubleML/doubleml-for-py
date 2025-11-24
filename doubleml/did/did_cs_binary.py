import warnings
from typing import Optional

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
    _check_score,
)
from doubleml.utils._estimation import _dml_cv_predict, _dml_tune, _get_cond_smpls_2d
from doubleml.utils._tune_optuna import _dml_tune_optuna
from doubleml.utils.propensity_score_processing import PSProcessorConfig, init_ps_processor


# TODO [v0.12.0]: Remove support for 'trimming_rule' and 'trimming_threshold' (deprecated).
class DoubleMLDIDCSBinary(LinearScoreMixin, DoubleML):
    """Double machine learning for difference-in-differences models with repeated cross sections
    (binary setting in terms of group and time combinations).

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
        Number of repetitions for the sample splitting.
        Default is ``1``.

    score : str
        A str (``'observational'`` or ``'experimental'``) specifying the score function.
        The ``'experimental'`` scores refers to an A/B setting, where the treatment is independent
        from the pretreatment covariates.
        Default is ``'observational'``.

    in_sample_normalization : bool
        Indicates whether to use a slightly different normalization from Sant'Anna and Zhao (2020).
        Default is ``True``.

    trimming_rule : str, optional, deprecated
        (DEPRECATED) A str (``'truncate'`` is the only choice) specifying the trimming approach.
        Use `ps_processor_config` instead. Will be removed in a future version.

    trimming_threshold : float, optional, deprecated
        (DEPRECATED) The threshold used for trimming.
        Use `ps_processor_config` instead. Will be removed in a future version.

    ps_processor_config : PSProcessorConfig, optional
        Configuration for propensity score processing (clipping, calibration, etc.).

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
        trimming_rule="truncate",  # TODO [v0.12.0]: Remove support for 'trimming_rule' and 'trimming_threshold' (deprecated).
        trimming_threshold=1e-2,  # TODO [v0.12.0]: Remove support for 'trimming_rule' and 'trimming_threshold' (deprecated).
        ps_processor_config: Optional[PSProcessorConfig] = None,
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
        self._data_subset = self._preprocess_data(self._g_value, self._t_value_pre, self._t_value_eval)

        # Handling id values to match pairwise evaluation & simultaneous inference
        if not np.all(np.isin(self.data_subset.index, self._dml_data.data.index)):
            raise ValueError("The index values in the data subset are not a subset of the original index values.")

        # Find position of data subset in original data
        # These entries should be replaced by nuisance predictions, all others should be set to 0.
        self._id_positions = self.data_subset.index.values

        # Numeric values for positions of the entries in id_panel_data inside id_original
        # np.nonzero(np.isin(id_original, id_panel_data))
        self._n_obs_subset = self.data_subset.shape[0]  # Effective sample size used for resampling

        # Save x and y for later ML estimation
        self._x_data_subset = self.data_subset.loc[:, self._dml_data.x_cols].values
        self._y_data_subset = self.data_subset.loc[:, self._dml_data.y_col].values
        self._g_data_subset = self.data_subset.loc[:, "G_indicator"].values
        self._t_data_subset = self.data_subset.loc[:, "t_indicator"].values

        valid_scores = ["observational", "experimental"]
        _check_score(self.score, valid_scores, allow_callable=False)

        self._in_sample_normalization = in_sample_normalization
        if not isinstance(self.in_sample_normalization, bool):
            raise TypeError(
                "in_sample_normalization indicator has to be boolean. "
                + f"Object of type {str(type(self.in_sample_normalization))} passed."
            )

        # set stratication for resampling
        self._strata = self.data_subset["G_indicator"] + 2 * self.data_subset["t_indicator"]
        self._n_obs_sample_splitting = self.n_obs_subset
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

        # TODO [v0.12.0]: Remove support for 'trimming_rule' and 'trimming_threshold' (deprecated).
        self._ps_processor_config, self._ps_processor = init_ps_processor(
            ps_processor_config, trimming_rule, trimming_threshold
        )
        self._trimming_rule = trimming_rule
        self._trimming_threshold = self._ps_processor.clipping_threshold

        self._sensitivity_implemented = True
        self._external_predictions_implemented = True

    def _format_score_info_str(self):
        lines = [
            f"Score function: {str(self.score)}",
            f"Treatment group: {str(self.g_value)}",
            f"Pre-treatment period: {str(self.t_value_pre)}",
            f"Evaluation period: {str(self.t_value_eval)}",
            f"Control group: {str(self.control_group)}",
            f"Anticipation periods: {str(self.anticipation_periods)}",
            f"Effective sample size: {str(self.n_obs_subset)}",
        ]
        return "\n".join(lines)

    # _format_learner_info_str method is inherited from DoubleML base class.

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
    def data_subset(self):
        """
        The preprocessed data subset.
        """
        return self._data_subset

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
    def ps_processor_config(self):
        """
        Configuration for propensity score processing (clipping, calibration, etc.).
        """
        return self._ps_processor_config

    @property
    def ps_processor(self):
        """
        Propensity score processor.
        """
        return self._ps_processor

    # TODO [v0.12.0]: Remove support for 'trimming_rule' and 'trimming_threshold' (deprecated).
    @property
    def trimming_rule(self):
        """
        Specifies the used trimming rule.
        """
        warnings.warn(
            "'trimming_rule' is deprecated and will be removed in a future version. ", DeprecationWarning, stacklevel=2
        )
        return self._trimming_rule

    # TODO [v0.12.0]: Remove support for 'trimming_rule' and 'trimming_threshold' (deprecated).
    @property
    def trimming_threshold(self):
        """
        Specifies the used trimming threshold.
        """
        warnings.warn(
            "'trimming_threshold' is deprecated and will be removed in a future version. "
            "Use 'ps_processor_config.clipping_threshold' or 'ps_processor.clipping_threshold' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._ps_processor.clipping_threshold

    @property
    def n_obs_subset(self):
        """
        The number of observations used for estimation.
        """
        return self._n_obs_subset

    def _initialize_ml_nuisance_params(self):
        if self.score == "observational":
            valid_learner = ["ml_g_d0_t0", "ml_g_d0_t1", "ml_g_d1_t0", "ml_g_d1_t1", "ml_m"]
        else:
            assert self.score == "experimental"
            valid_learner = ["ml_g_d0_t0", "ml_g_d0_t1", "ml_g_d1_t0", "ml_g_d1_t1"]
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

        t_col = self._dml_data.t_col
        id_col = self._dml_data.id_col
        g_col = self._dml_data.g_col

        # relevant data subset
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

        # add time indicator
        data_subset = data_subset.assign(t_indicator=data_subset[t_col] == eval_t)
        return data_subset

    def _estimate_conditional_g(
        self, x, y, d_val, t_val, d_arr, t_arr, smpls_cond, external_prediction, learner_param_key, n_jobs_cv, return_models
    ):
        """Helper function to estimate conditional g_hat for fixed d and t."""
        g_hat_cond = {}
        condition = (d_arr == d_val) & (t_arr == t_val)

        if external_prediction is not None:
            ml_g_targets = np.full_like(y, np.nan, dtype="float64")
            ml_g_targets[condition] = y[condition]
            ml_pred = _get_id_positions(external_prediction, self.id_positions)
            g_hat_cond = {"preds": ml_pred, "targets": ml_g_targets, "models": None}
        else:
            g_hat_cond = _dml_cv_predict(
                self._learner["ml_g"],
                x,
                y,
                smpls_cond,
                n_jobs=n_jobs_cv,
                est_params=self._get_params(learner_param_key),
                method=self._predict_method["ml_g"],
                return_models=return_models,
            )
            _check_finite_predictions(g_hat_cond["preds"], self._learner["ml_g"], "ml_g", smpls_cond)
            g_hat_cond["targets"] = g_hat_cond["targets"].astype(float)
            g_hat_cond["targets"][~condition] = np.nan
        return g_hat_cond

    def _nuisance_est(self, smpls, n_jobs_cv, external_predictions, return_models=False):

        # Here: d is a binary treatment indicator
        x, y = check_X_y(X=self._x_data_subset, y=self._y_data_subset, ensure_all_finite=False)
        _, d = check_X_y(x, self._g_data_subset, ensure_all_finite=False)  # (d is the G_indicator)
        _, t = check_X_y(x, self._t_data_subset, ensure_all_finite=False)

        # THIS DIFFERS FROM THE PAPER due to stratified splitting this should be the same for each fold
        # nuisance estimates of the uncond. treatment prob.
        p_hat = np.full_like(d, d.mean(), dtype="float64")
        lambda_hat = np.full_like(t, t.mean(), dtype="float64")

        # nuisance g
        smpls_d0_t0, smpls_d0_t1, smpls_d1_t0, smpls_d1_t1 = _get_cond_smpls_2d(smpls, d, t)

        g_hat_d0_t0 = self._estimate_conditional_g(
            x, y, 0, 0, d, t, smpls_d0_t0, external_predictions["ml_g_d0_t0"], "ml_g_d0_t0", n_jobs_cv, return_models
        )
        g_hat_d0_t1 = self._estimate_conditional_g(
            x, y, 0, 1, d, t, smpls_d0_t1, external_predictions["ml_g_d0_t1"], "ml_g_d0_t1", n_jobs_cv, return_models
        )
        g_hat_d1_t0 = self._estimate_conditional_g(
            x, y, 1, 0, d, t, smpls_d1_t0, external_predictions["ml_g_d1_t0"], "ml_g_d1_t0", n_jobs_cv, return_models
        )
        g_hat_d1_t1 = self._estimate_conditional_g(
            x, y, 1, 1, d, t, smpls_d1_t1, external_predictions["ml_g_d1_t1"], "ml_g_d1_t1", n_jobs_cv, return_models
        )

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
            m_hat["preds"] = self._ps_processor.adjust_ps(m_hat["preds"], d, cv=smpls, learner_name="ml_m")

        psi_a, psi_b = self._score_elements(
            y,
            d,
            t,
            g_hat_d0_t0["preds"],
            g_hat_d0_t1["preds"],
            g_hat_d1_t0["preds"],
            g_hat_d1_t1["preds"],
            m_hat["preds"],
            p_hat,
            lambda_hat,
        )

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
                "ml_g_d0_t0": _set_id_positions(g_hat_d0_t0["preds"], fill_value=np.nan, **extend_kwargs),
                "ml_g_d0_t1": _set_id_positions(g_hat_d0_t1["preds"], fill_value=np.nan, **extend_kwargs),
                "ml_g_d1_t0": _set_id_positions(g_hat_d1_t0["preds"], fill_value=np.nan, **extend_kwargs),
                "ml_g_d1_t1": _set_id_positions(g_hat_d1_t1["preds"], fill_value=np.nan, **extend_kwargs),
                "ml_m": _set_id_positions(m_hat["preds"], fill_value=np.nan, **extend_kwargs),
            },
            "targets": {
                "ml_g_d0_t0": _set_id_positions(g_hat_d0_t0["targets"], fill_value=np.nan, **extend_kwargs),
                "ml_g_d0_t1": _set_id_positions(g_hat_d0_t1["targets"], fill_value=np.nan, **extend_kwargs),
                "ml_g_d1_t0": _set_id_positions(g_hat_d1_t0["targets"], fill_value=np.nan, **extend_kwargs),
                "ml_g_d1_t1": _set_id_positions(g_hat_d1_t1["targets"], fill_value=np.nan, **extend_kwargs),
                "ml_m": _set_id_positions(m_hat["targets"], fill_value=np.nan, **extend_kwargs),
            },
            "models": {
                "ml_g_d0_t0": g_hat_d0_t0["models"],
                "ml_g_d0_t1": g_hat_d0_t1["models"],
                "ml_g_d1_t0": g_hat_d1_t0["models"],
                "ml_g_d1_t1": g_hat_d1_t1["models"],
                "ml_m": m_hat["models"],
            },
        }

        return psi_elements, preds

    def _score_elements(self, y, d, t, g_hat_d0_t0, g_hat_d0_t1, g_hat_d1_t0, g_hat_d1_t1, m_hat, p_hat, lambda_hat):
        # calculate residuals
        resid_d0_t0 = y - g_hat_d0_t0
        resid_d0_t1 = y - g_hat_d0_t1
        resid_d1_t0 = y - g_hat_d1_t0
        resid_d1_t1 = y - g_hat_d1_t1

        d1t1 = np.multiply(d, t)
        d1t0 = np.multiply(d, 1.0 - t)
        d0t1 = np.multiply(1.0 - d, t)
        d0t0 = np.multiply(1.0 - d, 1.0 - t)

        if self.score == "observational":
            if self.in_sample_normalization:
                weight_psi_a = np.divide(d, np.mean(d))
                weight_g_d1_t1 = weight_psi_a
                weight_g_d1_t0 = -1.0 * weight_psi_a
                weight_g_d0_t1 = -1.0 * weight_psi_a
                weight_g_d0_t0 = weight_psi_a

                weight_resid_d1_t1 = np.divide(d1t1, np.mean(d1t1))
                weight_resid_d1_t0 = -1.0 * np.divide(d1t0, np.mean(d1t0))

                prop_weighting = np.divide(m_hat, 1.0 - m_hat)
                unscaled_d0_t1 = np.multiply(d0t1, prop_weighting)
                weight_resid_d0_t1 = -1.0 * np.divide(unscaled_d0_t1, np.mean(unscaled_d0_t1))

                unscaled_d0_t0 = np.multiply(d0t0, prop_weighting)
                weight_resid_d0_t0 = np.divide(unscaled_d0_t0, np.mean(unscaled_d0_t0))
            else:
                weight_psi_a = np.divide(d, p_hat)
                weight_g_d1_t1 = weight_psi_a
                weight_g_d1_t0 = -1.0 * weight_psi_a
                weight_g_d0_t1 = -1.0 * weight_psi_a
                weight_g_d0_t0 = weight_psi_a

                weight_resid_d1_t1 = np.divide(d1t1, np.multiply(p_hat, lambda_hat))
                weight_resid_d1_t0 = -1.0 * np.divide(d1t0, np.multiply(p_hat, 1.0 - lambda_hat))

                prop_weighting = np.divide(m_hat, 1.0 - m_hat)
                weight_resid_d0_t1 = -1.0 * np.multiply(np.divide(d0t1, np.multiply(p_hat, lambda_hat)), prop_weighting)
                weight_resid_d0_t0 = np.multiply(np.divide(d0t0, np.multiply(p_hat, 1.0 - lambda_hat)), prop_weighting)
        else:
            assert self.score == "experimental"
            if self.in_sample_normalization:
                weight_psi_a = np.ones_like(y)
                weight_g_d1_t1 = weight_psi_a
                weight_g_d1_t0 = -1.0 * weight_psi_a
                weight_g_d0_t1 = -1.0 * weight_psi_a
                weight_g_d0_t0 = weight_psi_a

                weight_resid_d1_t1 = np.divide(d1t1, np.mean(d1t1))
                weight_resid_d1_t0 = -1.0 * np.divide(d1t0, np.mean(d1t0))
                weight_resid_d0_t1 = -1.0 * np.divide(d0t1, np.mean(d0t1))
                weight_resid_d0_t0 = np.divide(d0t0, np.mean(d0t0))
            else:
                weight_psi_a = np.ones_like(y)
                weight_g_d1_t1 = weight_psi_a
                weight_g_d1_t0 = -1.0 * weight_psi_a
                weight_g_d0_t1 = -1.0 * weight_psi_a
                weight_g_d0_t0 = weight_psi_a

                weight_resid_d1_t1 = np.divide(d1t1, np.multiply(p_hat, lambda_hat))
                weight_resid_d1_t0 = -1.0 * np.divide(d1t0, np.multiply(p_hat, 1.0 - lambda_hat))
                weight_resid_d0_t1 = -1.0 * np.divide(d0t1, np.multiply(1.0 - p_hat, lambda_hat))
                weight_resid_d0_t0 = np.divide(d0t0, np.multiply(1.0 - p_hat, 1.0 - lambda_hat))

        # set score elements
        psi_a = -1.0 * weight_psi_a

        # psi_b
        psi_b_1 = (
            np.multiply(weight_g_d1_t1, g_hat_d1_t1)
            + np.multiply(weight_g_d1_t0, g_hat_d1_t0)
            + np.multiply(weight_g_d0_t0, g_hat_d0_t0)
            + np.multiply(weight_g_d0_t1, g_hat_d0_t1)
        )
        psi_b_2 = (
            np.multiply(weight_resid_d1_t1, resid_d1_t1)
            + np.multiply(weight_resid_d1_t0, resid_d1_t0)
            + np.multiply(weight_resid_d0_t0, resid_d0_t0)
            + np.multiply(weight_resid_d0_t1, resid_d0_t1)
        )

        psi_b = psi_b_1 + psi_b_2

        return psi_a, psi_b

    def _nuisance_tuning(
        self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search
    ):
        x, y = check_X_y(X=self._x_data_subset, y=self._y_data_subset, ensure_all_finite=False)
        _, d = check_X_y(x, self._g_data_subset, ensure_all_finite=False)  # (d is the G_indicator)
        _, t = check_X_y(x, self._t_data_subset, ensure_all_finite=False)

        if scoring_methods is None:
            scoring_methods = {"ml_g": None, "ml_m": None}

        # nuisance training sets conditional on d and t
        smpls_d0_t0, smpls_d0_t1, smpls_d1_t0, smpls_d1_t1 = _get_cond_smpls_2d(smpls, d, t)
        train_inds = [train_index for (train_index, _) in smpls]
        train_inds_d0_t0 = [train_index for (train_index, _) in smpls_d0_t0]
        train_inds_d0_t1 = [train_index for (train_index, _) in smpls_d0_t1]
        train_inds_d1_t0 = [train_index for (train_index, _) in smpls_d1_t0]
        train_inds_d1_t1 = [train_index for (train_index, _) in smpls_d1_t1]

        tune_args = {
            "n_folds_tune": n_folds_tune,
            "n_jobs_cv": n_jobs_cv,
            "search_mode": search_mode,
            "n_iter_randomized_search": n_iter_randomized_search,
        }

        g_d0_t0_tune_res = _dml_tune(
            y,
            x,
            train_inds_d0_t0,
            self._learner["ml_g"],
            param_grids["ml_g"],
            scoring_methods["ml_g"],
            **tune_args,
        )

        g_d0_t1_tune_res = _dml_tune(
            y,
            x,
            train_inds_d0_t1,
            self._learner["ml_g"],
            param_grids["ml_g"],
            scoring_methods["ml_g"],
            **tune_args,
        )

        g_d1_t0_tune_res = _dml_tune(
            y,
            x,
            train_inds_d1_t0,
            self._learner["ml_g"],
            param_grids["ml_g"],
            scoring_methods["ml_g"],
            **tune_args,
        )

        g_d1_t1_tune_res = _dml_tune(
            y,
            x,
            train_inds_d1_t1,
            self._learner["ml_g"],
            param_grids["ml_g"],
            scoring_methods["ml_g"],
            **tune_args,
        )

        m_tune_res = list()
        if self.score == "observational":
            m_tune_res = _dml_tune(
                d,
                x,
                train_inds,
                self._learner["ml_m"],
                param_grids["ml_m"],
                scoring_methods["ml_m"],
                **tune_args,
            )

        g_d0_t0_best_params = [xx.best_params_ for xx in g_d0_t0_tune_res]
        g_d0_t1_best_params = [xx.best_params_ for xx in g_d0_t1_tune_res]
        g_d1_t0_best_params = [xx.best_params_ for xx in g_d1_t0_tune_res]
        g_d1_t1_best_params = [xx.best_params_ for xx in g_d1_t1_tune_res]

        if self.score == "observational":
            m_best_params = [xx.best_params_ for xx in m_tune_res]
            params = {
                "ml_g_d0_t0": g_d0_t0_best_params,
                "ml_g_d0_t1": g_d0_t1_best_params,
                "ml_g_d1_t0": g_d1_t0_best_params,
                "ml_g_d1_t1": g_d1_t1_best_params,
                "ml_m": m_best_params,
            }
            tune_res = {
                "g_d0_t0_tune": g_d0_t0_tune_res,
                "g_d0_t1_tune": g_d0_t1_tune_res,
                "g_d1_t0_tune": g_d1_t0_tune_res,
                "g_d1_t1_tune": g_d1_t1_tune_res,
                "m_tune": m_tune_res,
            }
        else:
            params = {
                "ml_g_d0_t0": g_d0_t0_best_params,
                "ml_g_d0_t1": g_d0_t1_best_params,
                "ml_g_d1_t0": g_d1_t0_best_params,
                "ml_g_d1_t1": g_d1_t1_best_params,
            }
            tune_res = {
                "g_d0_t0_tune": g_d0_t0_tune_res,
                "g_d0_t1_tune": g_d0_t1_tune_res,
                "g_d1_t0_tune": g_d1_t0_tune_res,
                "g_d1_t1_tune": g_d1_t1_tune_res,
            }

        res = {"params": params, "tune_res": tune_res}

        return res

    def _nuisance_tuning_optuna(
        self,
        optuna_params,
        scoring_methods,
        cv,
        optuna_settings,
    ):

        x, y = check_X_y(self._x_data_subset, self._y_data_subset, ensure_all_finite=False)
        _, d = check_X_y(x, self._g_data_subset, ensure_all_finite=False)
        _, t = check_X_y(x, self._t_data_subset, ensure_all_finite=False)

        if scoring_methods is None:
            if self.score == "observational":
                scoring_methods = {
                    "ml_g_d0_t0": None,
                    "ml_g_d0_t1": None,
                    "ml_g_d1_t0": None,
                    "ml_g_d1_t1": None,
                    "ml_m": None,
                }
            else:
                scoring_methods = {
                    "ml_g_d0_t0": None,
                    "ml_g_d0_t1": None,
                    "ml_g_d1_t0": None,
                    "ml_g_d1_t1": None,
                }

        masks = {
            "d0_t0": (d == 0) & (t == 0),
            "d0_t1": (d == 0) & (t == 1),
            "d1_t0": (d == 1) & (t == 0),
            "d1_t1": (d == 1) & (t == 1),
        }

        g_tune_results = {}
        for key, mask in masks.items():
            x_subset = x[mask, :]
            y_subset = y[mask]
            params_key = f"ml_g_{key}"
            param_grid = optuna_params[params_key]
            scoring = scoring_methods[params_key]
            g_tune_results[key] = _dml_tune_optuna(
                y_subset,
                x_subset,
                self._learner["ml_g"],
                param_grid,
                scoring,
                cv,
                optuna_settings,
                learner_name="ml_g",
                params_name=params_key,
            )

        m_tune_res = None
        if self.score == "observational":
            m_tune_res = _dml_tune_optuna(
                d,
                x,
                self._learner["ml_m"],
                optuna_params["ml_m"],
                scoring_methods["ml_m"],
                cv,
                optuna_settings,
                learner_name="ml_m",
                params_name="ml_m",
            )

        results = {f"ml_g_{key}": res_obj for key, res_obj in g_tune_results.items()}

        if self.score == "observational":
            results["ml_m"] = m_tune_res

        return results

    def _sensitivity_element_est(self, preds):
        y = self._y_data_subset
        d = self._g_data_subset
        t = self._t_data_subset

        m_hat = _get_id_positions(preds["predictions"]["ml_m"], self.id_positions)
        g_hat_d0_t0 = _get_id_positions(preds["predictions"]["ml_g_d0_t0"], self.id_positions)
        g_hat_d0_t1 = _get_id_positions(preds["predictions"]["ml_g_d0_t1"], self.id_positions)
        g_hat_d1_t0 = _get_id_positions(preds["predictions"]["ml_g_d1_t0"], self.id_positions)
        g_hat_d1_t1 = _get_id_positions(preds["predictions"]["ml_g_d1_t1"], self.id_positions)

        d0t0 = np.multiply(1.0 - d, 1.0 - t)
        d0t1 = np.multiply(1.0 - d, t)
        d1t0 = np.multiply(d, 1.0 - t)
        d1t1 = np.multiply(d, t)

        g_hat = (
            np.multiply(d0t0, g_hat_d0_t0)
            + np.multiply(d0t1, g_hat_d0_t1)
            + np.multiply(d1t0, g_hat_d1_t0)
            + np.multiply(d1t1, g_hat_d1_t1)
        )
        sigma2_score_element = np.square(y - g_hat)
        sigma2 = np.mean(sigma2_score_element)
        psi_sigma2 = sigma2_score_element - sigma2

        # calc m(W,alpha) and Riesz representer
        p_hat = np.mean(d)
        lambda_hat = np.mean(t)
        if self.score == "observational":
            propensity_weight_d0 = np.divide(m_hat, 1.0 - m_hat)
            if self.in_sample_normalization:
                weight_d0t1 = np.multiply(d0t1, propensity_weight_d0)
                weight_d0t0 = np.multiply(d0t0, propensity_weight_d0)
                mean_weight_d0t1 = np.mean(weight_d0t1)
                mean_weight_d0t0 = np.mean(weight_d0t0)

                m_alpha = np.multiply(
                    np.divide(d, p_hat),
                    np.divide(1.0, np.mean(d1t1))
                    + np.divide(1.0, np.mean(d1t0))
                    + np.divide(propensity_weight_d0, mean_weight_d0t1)
                    + np.divide(propensity_weight_d0, mean_weight_d0t0),
                )

                rr = (
                    np.divide(d1t1, np.mean(d1t1))
                    - np.divide(d1t0, np.mean(d1t0))
                    - np.divide(weight_d0t1, mean_weight_d0t1)
                    + np.divide(weight_d0t0, mean_weight_d0t0)
                )
            else:
                m_alpha_1 = np.divide(1.0, lambda_hat) + np.divide(1.0, 1.0 - lambda_hat)
                m_alpha = np.multiply(np.divide(d, np.square(p_hat)), np.multiply(m_alpha_1, 1.0 + propensity_weight_d0))

                rr_1 = np.divide(t, np.multiply(p_hat, lambda_hat)) + np.divide(1.0 - t, np.multiply(p_hat, 1.0 - lambda_hat))
                rr_2 = d + np.multiply(1.0 - d, propensity_weight_d0)
                rr = np.multiply(rr_1, rr_2)
        else:
            assert self.score == "experimental"
            if self.in_sample_normalization:
                m_alpha = (
                    np.divide(1.0, np.mean(d1t1))
                    + np.divide(1.0, np.mean(d1t0))
                    + np.divide(1.0, np.mean(d0t1))
                    + np.divide(1.0, np.mean(d0t0))
                )
                rr = (
                    np.divide(d1t1, np.mean(d1t1))
                    - np.divide(d1t0, np.mean(d1t0))
                    - np.divide(d0t1, np.mean(d0t1))
                    + np.divide(d0t0, np.mean(d0t0))
                )
            else:
                m_alpha = (
                    np.divide(1.0, np.multiply(p_hat, lambda_hat))
                    + np.divide(1.0, np.multiply(p_hat, 1.0 - lambda_hat))
                    + np.divide(1.0, np.multiply(1.0 - p_hat, lambda_hat))
                    + np.divide(1.0, np.multiply(1.0 - p_hat, 1.0 - lambda_hat))
                )
                rr = (
                    np.divide(d1t1, np.multiply(p_hat, lambda_hat))
                    - np.divide(d1t0, np.multiply(p_hat, 1.0 - lambda_hat))
                    - np.divide(d0t1, np.multiply(1.0 - p_hat, lambda_hat))
                    + np.divide(d0t0, np.multiply(1.0 - p_hat, 1.0 - lambda_hat))
                )

        nu2_score_element = np.multiply(2.0, m_alpha) - np.square(rr)
        nu2 = np.mean(nu2_score_element)
        psi_nu2 = nu2_score_element - nu2

        extend_kwargs = {
            "n_obs": self._dml_data.n_obs,
            "id_positions": self.id_positions,
            "fill_value": 0.0,
        }

        # add scaling to make variance estimation consistent (sample size difference)
        scaling = self._dml_data.n_obs / self._n_obs_subset
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
