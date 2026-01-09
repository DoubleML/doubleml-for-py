import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from sklearn.utils import check_X_y

from doubleml.data.panel_data import DoubleMLPanelData
from doubleml.double_ml import DoubleML
from doubleml.double_ml_score_mixins import LinearScoreMixin
from doubleml.utils._checks import _check_binary_predictions, _check_finite_predictions, _check_is_propensity, _check_score
from doubleml.utils._estimation import _dml_cv_predict, _dml_tune
from doubleml.utils._tune_optuna import _dml_tune_optuna


class DoubleMLPLPR(LinearScoreMixin, DoubleML):
    """Double machine learning for partially linear panel regression models

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLPanelData` object
        The :class:`DoubleMLPanelData` object providing the data and specifying the variables for the causal model.

    ml_l : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`\\ell_0(X) = E[Y|X]`.

    ml_m : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`m_0(X) = E[D|X]`.
        For binary treatment variables :math:`D` (with values 0 and 1) and the CRE approaches, a classifier
        implementing ``fit()`` and ``predict_proba()`` can also be specified. If :py:func:`sklearn.base.is_classifier`
        returns ``True``, ``predict_proba()`` is used otherwise ``predict()``.

    ml_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function
        :math:`g_0(X) = E[Y - D \\theta_0|X]`.
        Note: The learner `ml_g` is only required for the score ``'IV-type'``.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitons for the sample splitting.
        Default is ``1``.

    score : str
        A str (``'partialling out'`` or ``'IV-type'``) specifying the score function.
        Default is ``'partialling out'``.

    approach : str
        A str (``'cre_general'``, ``'cre_normal'``, ``'fd_exact'``, ``'wg_approx'``) specifying the type of
        static panel approach in Clarke and Polselli (2025). ``'cre_general'`` indicates the correlated random
        effect approach in the general case, while ``'cre_normal'`` assumes that the conditional distribution
        :math:`D_{i1}, \\dots, D_{iT} | X_{i1}, \\dots X_{iT}` is multivariate normal. ``'fd_exact'`` for the
        first-difference transformation exact approach, and ``'wg_approx'`` for the within-group transformation
        approximate approach.
        Default is ``'fd_exact'``.

    draw_sample_splitting : bool
        Indicates whether the sample splitting should be drawn during initialization of the object.
        Default is ``True``.

    Examples
    --------
    >>> import numpy as np
    >>> import doubleml as dml
    >>> from doubleml.plm.datasets import make_plpr_CP2025
    >>> from sklearn.linear_model import LassoCV
    >>> from sklearn.base import clone
    >>> np.random.seed(3142)
    >>> learner = LassoCV()
    >>> ml_l = clone(learner)
    >>> ml_m = clone(learner)
    >>> data = make_plpr_CP2025(num_id=250, num_t=10, dim_x=30, theta=0.5, dgp_type='dgp1')
    >>> obj_dml_data = DoubleMLPanelData(data, 'y', 'd', 'time', 'id', static_panel=True)
    >>> dml_plpr_obj = DoubleMLPLPR(obj_dml_data, ml_l, ml_m)
    >>> dml_plpr_obj.fit().summary  # doctest: +SKIP
                coef   std err          t	      P>|t|	    2.5 %	 97.5 %
    d_diff  0.511626  0.024615  20.784933  5.924636e-96  0.463381  0.559871

    Notes
    -----
    **Partially linear panel regression (PLPR)** models take the form

    .. math::

        Y_{it} &= D_{it} \\theta_0 + g_0(X_{it}) + \\alpha_i^* + U_{it}, & &\\mathbb{E}(U_{it} | D_{it},X_{it},\\alpha_i) = 0,

        D_{it} &= m_0(X_{it}) + \\gamma_i + V_{it}, & &\\mathbb{E}(V_{it} | X_{it},\\gamma_i) = 0,

    where :math:`Y_{it}` is the outcome variable and :math:`D_{it}` is the policy variable of interest.
    The high-dimensional vector :math:`X_{it} = (X_{it,1}, \\ldots, X_{it,p})` consists of other confounding covariates,
    :math:`\\alpha_i^*` and :math:`\\gamma_i` are the unobserved individual heterogeneity correlated with the included
    covariates, and :math:`U_{it}` and :math:`V_{it}` are stochastic errors.
    """

    def __init__(
        self,
        obj_dml_data,
        ml_l,
        ml_m,
        ml_g=None,
        n_folds=5,
        n_rep=1,
        score="partialling out",
        approach="fd_exact",
        draw_sample_splitting=True,
    ):
        self._check_data(obj_dml_data)
        self._original_dml_data = obj_dml_data

        valid_approach = ["cre_general", "cre_normal", "fd_exact", "wg_approx"]
        self._check_approach(approach, valid_approach)
        self._approach = approach

        # pass transformed data as DoubleMLPanelData to init
        data_transform, self._transform_cols = self._transform_data()
        obj_dml_data_transform = DoubleMLPanelData(
            data_transform,
            y_col=self._transform_cols["y_col"],
            d_cols=self._transform_cols["d_cols"],
            t_col=self._original_dml_data._t_col,
            id_col=self._original_dml_data._id_col,
            x_cols=self._transform_cols["x_cols"],
            z_cols=self._original_dml_data._z_cols,
            static_panel=True,
            use_other_treat_as_covariate=self._original_dml_data._use_other_treat_as_covariate,
            force_all_x_finite=self._original_dml_data._force_all_x_finite,
        )
        super().__init__(obj_dml_data_transform, n_folds, n_rep, score, draw_sample_splitting)

        valid_scores = ["IV-type", "partialling out"]
        _check_score(self.score, valid_scores, allow_callable=False)
        _ = self._check_learner(ml_l, "ml_l", regressor=True, classifier=False)
        ml_m_is_classifier = self._check_learner(ml_m, "ml_m", regressor=True, classifier=True)
        self._learner = {"ml_l": ml_l, "ml_m": ml_m}

        if ml_g is not None:
            if self.score == "IV-type":
                _ = self._check_learner(ml_g, "ml_g", regressor=True, classifier=False)
                self._learner["ml_g"] = ml_g
            else:
                assert self.score == "partialling out"
                warnings.warn(
                    (
                        'A learner ml_g has been provided for score = "partialling out" but will be ignored. "'
                        "A learner ml_g is not required for estimation."
                    )
                )
        elif self.score == "IV-type":
            warnings.warn(("For score = 'IV-type', learners ml_l and ml_g should be specified. Set ml_g = clone(ml_l)."))
            self._learner["ml_g"] = clone(ml_l)

        self._predict_method = {"ml_l": "predict"}
        if "ml_g" in self._learner:
            self._predict_method["ml_g"] = "predict"
        if ml_m_is_classifier:
            if self._dml_data.binary_treats.all():
                self._predict_method["ml_m"] = "predict_proba"
            else:
                msg = (
                    f"The ml_m learner {str(ml_m)} was identified as classifier "
                    "but at least one treatment variable is not binary with values 0 and 1."
                )
                if self._approach in ["fd_exact", "wg_approx"]:
                    msg += (
                        " Note: In case of binary input treatment variable(s), approaches 'fd_exact' and "
                        "'wg_approx' tansform the treatment variable(s), such that they are no longer binary."
                    )
                raise ValueError(msg)
        else:
            self._predict_method["ml_m"] = "predict"

        self._initialize_ml_nuisance_params()
        self._sensitivity_implemented = False
        self._external_predictions_implemented = True
        self._set_d_mean()

    def _format_score_info_str(self):
        score_approach_info = f"Score function: {str(self.score)}\n" f"Static panel model approach: {str(self.approach)}"
        return score_approach_info

    def _format_additional_info_str(self):
        """
        Includes information on the original data before transformation.
        """
        data_original_summary = (
            f"Cluster variable(s): {self._original_dml_data.cluster_cols}\n"
            f"\nPre-Transformation Data Summary: \n"
            f"Outcome variable: {self._original_dml_data.y_col}\n"
            f"Treatment variable(s): {self._original_dml_data.d_cols}\n"
            f"Covariates: {self._original_dml_data.x_cols}\n"
            f"No. Observations: {self._original_dml_data.n_obs}\n"
        )
        return data_original_summary

    @property
    def approach(self):
        """
        The static panel approach.
        """
        return self._approach

    @property
    def data_transform(self):
        """
        The transformed static panel data object.
        """
        return self._dml_data

    @property
    def data_original(self):
        """
        The original static panel data object.
        """
        return self._original_dml_data

    @property
    def transform_cols(self):
        """
        The column names of the transformed static panel data object.
        """
        return self._transform_cols

    @property
    def d_mean(self):
        """
        The group mean of the treatment used for approaches ``'cre_general'`` and ``'cre_normal'``.
        ``None`` for approaches ``'fd_exact'``, ``'wg_approx'``.
        """
        return self._d_mean

    def _initialize_ml_nuisance_params(self):
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in self._learner}

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLPanelData):
            raise TypeError(f"The data must be of DoubleMLPanelData type. {str(type(obj_dml_data))} was passed.")
        if not obj_dml_data.static_panel:
            raise ValueError(
                "For the PLPR model, the DoubleMLPanelData object requires the static_panel flag to be set to True."
            )
        if obj_dml_data.z_cols is not None:
            raise ValueError(
                "Incompatible data. " + " and ".join(obj_dml_data.z_cols) + " have been set as instrumental variable(s). "
                "DoubleMLPLPR currently does not support instrumental variables."
            )
        return

    def _check_approach(self, approach, valid_approach):
        if isinstance(approach, str):
            if approach not in valid_approach:
                raise ValueError("Invalid approach " + approach + ". " + "Valid approach " + " or ".join(valid_approach) + ".")
        else:
            raise TypeError(f"approach should be a string. {str(approach)} was passed.")
        return

    def _transform_data(self):
        y_col = self._original_dml_data.y_col
        d_cols = self._original_dml_data.d_cols
        x_cols = self._original_dml_data.x_cols
        t_col = self._original_dml_data.t_col
        id_col = self._original_dml_data.id_col

        df = self._original_dml_data.data.sort_values([id_col, t_col]).copy()

        # check for ids with only one row
        id_counts = df[id_col].value_counts()
        single_row_ids = id_counts[id_counts == 1].index
        if not single_row_ids.empty:
            warnings.warn(
                f"The data contains {len(single_row_ids)} id(s) with only one row. These row(s) have been dropped.",
            )
            df = df[~df[id_col].isin(single_row_ids)].reset_index(drop=True)

        if self._approach in ["cre_general", "cre_normal"]:
            df_id_means = df[[id_col] + x_cols].groupby(id_col).transform("mean")
            df_means = df_id_means.add_suffix("_mean")
            data = pd.concat([df, df_means], axis=1)
            cols = {"y_col": y_col, "d_cols": d_cols, "x_cols": x_cols + [f"{x}_mean" for x in x_cols]}
        elif self._approach == "fd_exact":
            # potential issues with unbalanced panels/missing periods. Reindex to a complete time grid per id.

            # all unique ids and time periods
            ids = df[id_col].unique()
            # sort unique times to ensure correct order after reindexing
            times = np.sort(df[t_col].unique())
            # build multiIndex
            full_index = pd.MultiIndex.from_product([ids, times], names=[id_col, t_col])
            current_index = pd.MultiIndex.from_frame(df[[id_col, t_col]])
            missing_time_periods = full_index.difference(current_index)
            if len(missing_time_periods) > 0:
                warnings.warn(
                    (
                        f"The panel data contains {len(missing_time_periods)} missing (id, time) combinations. "
                        "Missing periods have been inserted with NaN values. As a consequence, first-difference "
                        "and lagged variables for these periods will also be NaN, and these rows will be dropped."
                    )
                )
            # reindex, insert missing rows with NaN
            df = df.set_index([id_col, t_col]).reindex(full_index).reset_index()
            shifted = df[[id_col] + x_cols].groupby(id_col).shift(1).add_suffix("_lag")
            first_diff = df[[id_col] + [y_col] + d_cols].groupby(id_col).diff().add_suffix("_diff")
            df_fd = pd.concat([df, shifted], axis=1)
            # replace original y and d columns for first-difference transformations, rename
            df_fd[[y_col] + d_cols] = first_diff
            cols_rename_dict = {y_col: f"{y_col}_diff"} | {col: f"{col}_diff" for col in d_cols}
            df_fd = df_fd.rename(columns=cols_rename_dict)
            # drop missing-panel rows and first periods: if either x or x_lag is missing
            data = df_fd.dropna(subset=[x_cols[0], x_cols[0] + "_lag"]).reset_index(drop=True)
            cols = {
                "y_col": f"{y_col}_diff",
                "d_cols": [f"{d}_diff" for d in d_cols],
                "x_cols": x_cols + [f"{x}_lag" for x in x_cols],
            }
        elif self._approach == "wg_approx":
            cols_to_demean = [y_col] + d_cols + x_cols
            # compute group and grand means for within means
            group_means = df.groupby(id_col)[cols_to_demean].transform("mean")
            grand_means = df[cols_to_demean].mean()
            within_means = df[cols_to_demean] - group_means + grand_means
            within_means = within_means.add_suffix("_demean")
            data = pd.concat([df[[id_col, t_col]], within_means], axis=1)
            cols = {
                "y_col": f"{y_col}_demean",
                "d_cols": [f"{d}_demean" for d in d_cols],
                "x_cols": [f"{x}_demean" for x in x_cols],
            }

        return data, cols

    def _set_d_mean(self):
        if self._approach in ["cre_general", "cre_normal"]:
            data = self._original_dml_data.data
            d_cols = self._original_dml_data.d_cols
            id_col = self._original_dml_data.id_col
            help_d_mean = data.loc[:, [id_col] + d_cols]
            d_mean = help_d_mean.groupby(id_col).transform("mean").values
            self._d_mean = d_mean
        else:
            self._d_mean = None

    def _nuisance_est(self, smpls, n_jobs_cv, external_predictions, return_models=False):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y, ensure_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d, ensure_all_finite=False)
        m_external = external_predictions["ml_m"] is not None
        l_external = external_predictions["ml_l"] is not None
        if "ml_g" in self._learner:
            g_external = external_predictions["ml_g"] is not None
        else:
            g_external = False

        # nuisance l
        if l_external:
            l_hat = {"preds": external_predictions["ml_l"], "targets": None, "models": None}
        elif self._score == "IV-type" and g_external:
            l_hat = {"preds": None, "targets": None, "models": None}
        else:
            l_hat = _dml_cv_predict(
                self._learner["ml_l"],
                x,
                y,
                smpls=smpls,
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_l"),
                method=self._predict_method["ml_l"],
                return_models=return_models,
            )
            _check_finite_predictions(l_hat["preds"], self._learner["ml_l"], "ml_l", smpls)

        # nuisance m
        if m_external:
            m_hat = {"preds": external_predictions["ml_m"], "targets": None, "models": None}
        else:
            if self._approach == "cre_normal":
                d_mean = self._d_mean[:, self._i_treat]
                x_m = np.column_stack((x, d_mean))
            else:
                x_m = x

            m_hat = _dml_cv_predict(
                self._learner["ml_m"],
                x_m,
                d,
                smpls=smpls,
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_m"),
                method=self._predict_method["ml_m"],
                return_models=return_models,
            )

            # general cre adjustment
            if self._approach == "cre_general":
                d_mean = self._d_mean[:, self._i_treat]
                df_m_hat = pd.DataFrame({"id": self._dml_data.id_var, "m_hat": m_hat["preds"]})
                m_hat_mean = df_m_hat.groupby(["id"]).transform("mean")
                m_hat_star = m_hat["preds"] + d_mean - m_hat_mean["m_hat"]
                m_hat["preds"] = m_hat_star

            _check_finite_predictions(m_hat["preds"], self._learner["ml_m"], "ml_m", smpls)
        if self._check_learner(self._learner["ml_m"], "ml_m", regressor=True, classifier=True):
            _check_is_propensity(m_hat["preds"], self._learner["ml_m"], "ml_m", smpls, eps=1e-12)

        if self._dml_data.binary_treats[self._dml_data.d_cols[self._i_treat]]:
            _check_binary_predictions(m_hat["preds"], self._learner["ml_m"], "ml_m", self._dml_data.d_cols[self._i_treat])

        # an estimate of g is obtained for the IV-type score
        g_hat = {"preds": None, "targets": None, "models": None}
        if "ml_g" in self._learner:
            # nuisance g
            if g_external:
                g_hat = {"preds": external_predictions["ml_g"], "targets": None, "models": None}
            else:
                # get an initial estimate for theta using the partialling out score
                psi_a = -np.multiply(d - m_hat["preds"], d - m_hat["preds"])
                psi_b = np.multiply(d - m_hat["preds"], y - l_hat["preds"])
                theta_initial = -np.nanmean(psi_b) / np.nanmean(psi_a)
                g_hat = _dml_cv_predict(
                    self._learner["ml_g"],
                    x,
                    y - theta_initial * d,
                    smpls=smpls,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_g"),
                    method=self._predict_method["ml_g"],
                    return_models=return_models,
                )
                _check_finite_predictions(g_hat["preds"], self._learner["ml_g"], "ml_g", smpls)

        psi_a, psi_b = self._score_elements(y, d, l_hat["preds"], m_hat["preds"], g_hat["preds"], smpls)
        psi_elements = {"psi_a": psi_a, "psi_b": psi_b}
        preds = {
            "predictions": {"ml_l": l_hat["preds"], "ml_m": m_hat["preds"], "ml_g": g_hat["preds"]},
            "targets": {"ml_l": l_hat["targets"], "ml_m": m_hat["targets"], "ml_g": g_hat["targets"]},
            "models": {"ml_l": l_hat["models"], "ml_m": m_hat["models"], "ml_g": g_hat["models"]},
        }

        return psi_elements, preds

    def _score_elements(self, y, d, l_hat, m_hat, g_hat, smpls):
        # compute residual
        v_hat = d - m_hat

        if self.score == "IV-type":
            psi_a = -np.multiply(v_hat, d)
            psi_b = np.multiply(v_hat, y - g_hat)
        else:
            assert self.score == "partialling out"
            u_hat = y - l_hat
            psi_a = -np.multiply(v_hat, v_hat)
            psi_b = np.multiply(v_hat, u_hat)

        return psi_a, psi_b

    def _sensitivity_element_est(self, preds):
        pass

    def _nuisance_tuning(
        self,
        smpls,
        param_grids,
        scoring_methods,
        n_folds_tune,
        n_jobs_cv,
        search_mode,
        n_iter_randomized_search,
    ):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y, ensure_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d, ensure_all_finite=False)

        if scoring_methods is None:
            scoring_methods = {"ml_l": None, "ml_m": None, "ml_g": None}

        train_inds = [train_index for (train_index, _) in smpls]
        l_tune_res = _dml_tune(
            y,
            x,
            train_inds,
            self._learner["ml_l"],
            param_grids["ml_l"],
            scoring_methods["ml_l"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )
        if self._approach == "cre_normal":
            d_mean = self._d_mean[:, self._i_treat]
            x_m = np.column_stack((x, d_mean))
        else:
            x_m = x

        m_tune_res = _dml_tune(
            d,
            x_m,
            train_inds,
            self._learner["ml_m"],
            param_grids["ml_m"],
            scoring_methods["ml_m"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )

        l_best_params = [xx.best_params_ for xx in l_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]

        # an ML model for g is obtained for the IV-type score
        if "ml_g" in self._learner:
            # construct an initial theta estimate from the tuned models using the partialling out score
            l_hat = np.full_like(y, np.nan)
            m_hat = np.full_like(d, np.nan)
            for idx, (train_index, _) in enumerate(smpls):
                l_hat[train_index] = l_tune_res[idx].predict(x[train_index, :])
                m_hat[train_index] = m_tune_res[idx].predict(x_m[train_index, :])
            psi_a = -np.multiply(d - m_hat, d - m_hat)
            psi_b = np.multiply(d - m_hat, y - l_hat)
            theta_initial = -np.nanmean(psi_b) / np.nanmean(psi_a)
            g_tune_res = _dml_tune(
                y - theta_initial * d,
                x,
                train_inds,
                self._learner["ml_g"],
                param_grids["ml_g"],
                scoring_methods["ml_g"],
                n_folds_tune,
                n_jobs_cv,
                search_mode,
                n_iter_randomized_search,
            )

            g_best_params = [xx.best_params_ for xx in g_tune_res]
            params = {"ml_l": l_best_params, "ml_m": m_best_params, "ml_g": g_best_params}
            tune_res = {"l_tune": l_tune_res, "m_tune": m_tune_res, "g_tune": g_tune_res}
        else:
            params = {"ml_l": l_best_params, "ml_m": m_best_params}
            tune_res = {"l_tune": l_tune_res, "m_tune": m_tune_res}

        res = {"params": params, "tune_res": tune_res}

        return res

    def _nuisance_tuning_optuna(
        self,
        optuna_params,
        scoring_methods,
        cv,
        optuna_settings,
    ):
        """
        Optuna-based hyperparameter tuning for PLPR nuisance models.

        Performs tuning once on the whole dataset using cross-validation,
        returning the same optimal parameters for all folds.
        """
        x, y = check_X_y(self._dml_data.x, self._dml_data.y, ensure_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d, ensure_all_finite=False)

        if scoring_methods is None:
            scoring_methods = {"ml_l": None, "ml_m": None, "ml_g": None}

        l_tune_res = _dml_tune_optuna(
            y,
            x,
            self._learner["ml_l"],
            optuna_params["ml_l"],
            scoring_methods["ml_l"],
            cv,
            optuna_settings,
            learner_name="ml_l",
            params_name="ml_l",
        )

        if self._approach == "cre_normal":
            d_mean = self._d_mean[:, self._i_treat]
            x_m = np.column_stack((x, d_mean))
        else:
            x_m = x
        m_tune_res = _dml_tune_optuna(
            d,
            x_m,
            self._learner["ml_m"],
            optuna_params["ml_m"],
            scoring_methods["ml_m"],
            cv,
            optuna_settings,
            learner_name="ml_m",
            params_name="ml_m",
        )

        results = {"ml_l": l_tune_res, "ml_m": m_tune_res}

        # an ML model for g is obtained for the IV-type score
        if "ml_g" in self._learner:
            # construct an initial theta estimate from the tuned models using the partialling out score
            # use cross-fitting for tuning ml_g
            l_hat = cross_val_predict(l_tune_res.best_estimator, x, y, cv=cv, method=self._predict_method["ml_l"])
            m_hat = cross_val_predict(m_tune_res.best_estimator, x_m, d, cv=cv, method=self._predict_method["ml_m"])
            psi_a = -np.multiply(d - m_hat, d - m_hat)
            psi_b = np.multiply(d - m_hat, y - l_hat)
            theta_initial = -np.nanmean(psi_b) / np.nanmean(psi_a)
            g_tune_res = _dml_tune_optuna(
                y - theta_initial * d,
                x,
                self._learner["ml_g"],
                optuna_params["ml_g"],
                scoring_methods["ml_g"],
                cv,
                optuna_settings,
                learner_name="ml_g",
                params_name="ml_g",
            )
            results["ml_g"] = g_tune_res

        return results
