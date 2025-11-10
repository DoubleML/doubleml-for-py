import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils import check_X_y

from ..data.panel_data import DoubleMLPanelData
from ..double_ml import DoubleML
from ..double_ml_score_mixins import LinearScoreMixin
from ..utils._checks import _check_binary_predictions, _check_finite_predictions, _check_is_propensity, _check_score
from ..utils._estimation import _dml_cv_predict, _dml_tune


class DoubleMLPLPR(LinearScoreMixin, DoubleML):
    """Double machine learning for partially linear panel regression models

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    ml_l : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`\\ell_0(X) = E[Y|X]`.

    ml_m : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`m_0(X) = E[D|X]`.
        For binary treatment variables :math:`D` (with values 0 and 1), a classifier implementing ``fit()`` and
        ``predict_proba()`` can also be specified. If :py:func:`sklearn.base.is_classifier` returns ``True``,
        ``predict_proba()`` is used otherwise ``predict()``.

    ml_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function
        :math:`g_0(X) = E[Y - D \\theta_0|X]`.
        Note: The learner `ml_g` is only required for the score ``'IV-type'``. Optionally, it can be specified and
        estimated for callable scores.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitons for the sample splitting.
        Default is ``1``.

    score : str or callable
        A str (``'partialling out'`` or ``'IV-type'``) specifying the score function
        or a callable object / function with signature ``psi_a, psi_b = score(y, d, l_hat, m_hat, g_hat, smpls)``.
        Default is ``'partialling out'``.

    static_panel_approach : str
        A str (``'cre_general'``, ``'cre_normal'``, ``'fd_exact'``, ``'wg_approx'``) specifying the type of
        static panel approach in Clarke and Polselli (2025).
        Default is ``'fd_exact'``.

    draw_sample_splitting : bool
        Indicates whether the sample splitting should be drawn during initialization of the object.
        Default is ``True``.

    TODO: include example and notes
    Examples
    --------
    >>> import numpy as np
    >>> import doubleml as dml
    

    Notes
    -----
    **Partially linear panel regression (PLPR)** models take the form

    """

    def __init__(
        self, obj_dml_data, ml_l, ml_m, ml_g=None, n_folds=5, n_rep=1, score="partialling out", static_panel_approach="fd_exact", draw_sample_splitting=True):
        super().__init__(obj_dml_data, n_folds, n_rep, score, draw_sample_splitting)

        self._check_data(self._dml_data)
        # TODO: assert cluster?
        valid_scores = ["IV-type", "partialling out"]
        _check_score(self.score, valid_scores, allow_callable=True)

        valid_static_panel_approach = ["cre_general", "cre_normal", "fd_exact", "wg_approx"]
        self._check_static_panel_approach(static_panel_approach, valid_static_panel_approach)
        self._static_panel_approach = static_panel_approach

        _ = self._check_learner(ml_l, "ml_l", regressor=True, classifier=False)
        ml_m_is_classifier = self._check_learner(ml_m, "ml_m", regressor=True, classifier=True)
        self._learner = {"ml_l": ml_l, "ml_m": ml_m}

        if ml_g is not None:
            if (isinstance(self.score, str) & (self.score == "IV-type")) | callable(self.score):
                _ = self._check_learner(ml_g, "ml_g", regressor=True, classifier=False)
                self._learner["ml_g"] = ml_g
            else:
                assert isinstance(self.score, str) & (self.score == "partialling out")
                warnings.warn(
                    (
                        'A learner ml_g has been provided for score = "partialling out" but will be ignored. "'
                        "A learner ml_g is not required for estimation."
                    )
                )
        elif isinstance(self.score, str) & (self.score == "IV-type"):
            warnings.warn(("For score = 'IV-type', learners ml_l and ml_g should be specified. Set ml_g = clone(ml_l)."))
            self._learner["ml_g"] = clone(ml_l)

        self._predict_method = {"ml_l": "predict"}
        if "ml_g" in self._learner:
            self._predict_method["ml_g"] = "predict"
        if ml_m_is_classifier:
            if self._dml_data.binary_treats.all():
                self._predict_method["ml_m"] = "predict_proba"
            else:
                raise ValueError(
                    f"The ml_m learner {str(ml_m)} was identified as classifier "
                    "but at least one treatment variable is not binary with values 0 and 1."
                )
        else:
            self._predict_method["ml_m"] = "predict"

        self._initialize_ml_nuisance_params()
        self._sensitivity_implemented = False
        self._external_predictions_implemented = True

        # Get transformed data depending on approach
        # TODO: get y, x, d cols, set additional properties for y_data, d_data, x_data to be used in
        # nuisance
        self._data_transform = self._transform_data(self._static_panel_approach)


    def _format_score_info_str(self):
        score_static_panel_approach_info = (
            f"Score function: {str(self.score)}\n"
            f"Static panel model approach: {str(self.static_panel_approach)}"
        )
        return score_static_panel_approach_info

    def _format_additional_info_str(self):
        """
        Includes information on the transformed features based on the estimation approach.
        """
        # TODO: Add Information on features after transformation
        return ""
    
    @property
    def static_panel_approach(self):
        """
        The score function.
        """
        return self._static_panel_approach

    @property
    def data_transform(self):
        """
        The transformed static panel data.
        """
        return self._data_transform

    def _initialize_ml_nuisance_params(self):
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in self._learner}

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLPanelData):
            raise TypeError(
                f"The data must be of DoubleMLPanelData type. {str(type(obj_dml_data))} was passed."
            )
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
    
    def _check_static_panel_approach(self, static_panel_approach, valid_static_panel_approach):
        if isinstance(static_panel_approach, str):
            if static_panel_approach not in valid_static_panel_approach:
                raise ValueError("Invalid static_panel_approach " + static_panel_approach + ". " + "Valid approach " + " or ".join(valid_static_panel_approach) + ".")
        else:
            raise TypeError(f"static_panel_approach should be a string. {str(static_panel_approach)} was passed.")
        return
    
    # TODO: preprocess and transform data based on static_panel_approach (cre, fd, wd)
    def _transform_data(self, static_panel_approach):
        df = self._dml_data.data.copy()

        y_col = self._dml_data.y_col
        d_cols = self._dml_data.d_cols
        x_cols = self._dml_data.x_cols
        t_col = self._dml_data.t_col
        id_col = self._dml_data.id_col

        if static_panel_approach in ["cre_general", "cre_normal"]:
            # uses regular y_col, d_cols, x_cols + m_x_cols
            df_id_means = df[[id_col] + d_cols + x_cols].groupby(id_col).transform("mean")
            df_means = df_id_means.add_prefix("mean_") 
            data = pd.concat([df, df_means], axis=1)
            # {"y_col": y_col, "d_cols": d_cols, "x_cols": x_cols + [f"m_{x}"" for x in x_cols]}
        elif static_panel_approach == "fd_exact":
            # TODO: potential issues with unbalanced panels/missing periods, right now the
            # last available is used as the lag and for diff. Maybe reindex to a complete time grid per id.
            # uses y_col_diff, d_cols_diff, x_cols + x_cols_lag
            df = df.sort_values([id_col, t_col])
            shifted = df[[id_col] + x_cols].groupby(id_col).shift(1).add_suffix("_lag")
            first_diff = df[[id_col] + d_cols + [y_col]].groupby(id_col).diff().add_suffix("_diff")
            df_fd = pd.concat([df, shifted, first_diff], axis=1)
            data = df_fd.dropna(subset=[x_cols[0] + "_lag"]).reset_index(drop=True)
            # {"y_col": f"{y_col}_diff", "d_cols": [f"{d}_diff" for d in d_cols], "x_cols": x_cols + [f"{x}_lag" for x in x_cols]}
        elif static_panel_approach == "wg_approx":
            # uses y_col, d_cols, x_cols
            df_demean = df.drop(t_col, axis=1).groupby(id_col).transform(lambda x: x - x.mean())
            # add grand means
            grand_means = df.drop([id_col, t_col], axis=1).mean()
            within_means = df_demean + grand_means
            data = pd.concat([df[[id_col, t_col]], within_means], axis=1)
            # {"y_col": y_col, "d_cols": d_cols, "x_cols": x_cols}
        else:
            raise ValueError(f"Invalid static_panel_approach.")

        return data

    def _nuisance_est(self, smpls, n_jobs_cv, external_predictions, return_models=False):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y, force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d, force_all_finite=False)
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
            # TODO: update this section
            # cre using m_d + x for m_hat, otherwise only x
            if self._static_panel_approach == "cre_normal":
                help_data = pd.DataFrame({"id": self._dml_data.cluster_vars[:, 0], "d": d})
                m_d = help_data.groupby(["id"]).transform("mean").values
                x = np.column_stack((x, m_d))

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

            # general cre adjustment
            if self._static_panel_approach == "cre_general":
                help_data = pd.DataFrame({"id": self._dml_data.cluster_vars[:, 0], "m_hat": m_hat["preds"], "d": d})
                group_means = help_data.groupby(["id"])[["m_hat", "d"]].transform("mean")
                m_hat_star = m_hat["preds"] + group_means["d"] - group_means["m_hat"]
                m_hat["preds"] = m_hat_star
                

            _check_finite_predictions(m_hat["preds"], self._learner["ml_m"], "ml_m", smpls)
        if self._check_learner(self._learner["ml_m"], "ml_m", regressor=True, classifier=True):
            _check_is_propensity(m_hat["preds"], self._learner["ml_m"], "ml_m", smpls, eps=1e-12)

        if self._dml_data.binary_treats[self._dml_data.d_cols[self._i_treat]]:
            _check_binary_predictions(m_hat["preds"], self._learner["ml_m"], "ml_m", self._dml_data.d_cols[self._i_treat])

        # an estimate of g is obtained for the IV-type score and callable scores
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

        if isinstance(self.score, str):
            if self.score == "IV-type":
                psi_a = -np.multiply(v_hat, d)
                psi_b = np.multiply(v_hat, y - g_hat)
            else:
                assert self.score == "partialling out"
                u_hat = y - l_hat
                psi_a = -np.multiply(v_hat, v_hat)
                psi_b = np.multiply(v_hat, u_hat)
        else:
            assert callable(self.score)
            psi_a, psi_b = self.score(y=y, d=d, l_hat=l_hat, m_hat=m_hat, g_hat=g_hat, smpls=smpls)

        return psi_a, psi_b

    def _sensitivity_element_est(self, preds):
        pass

    def _nuisance_tuning(
        self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search
    ):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y, force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d, force_all_finite=False)

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

        l_best_params = [xx.best_params_ for xx in l_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]

        # an ML model for g is obtained for the IV-type score and callable scores
        if "ml_g" in self._learner:
            # construct an initial theta estimate from the tuned models using the partialling out score
            l_hat = np.full_like(y, np.nan)
            m_hat = np.full_like(d, np.nan)
            for idx, (train_index, _) in enumerate(smpls):
                l_hat[train_index] = l_tune_res[idx].predict(x[train_index, :])
                m_hat[train_index] = m_tune_res[idx].predict(x[train_index, :])
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