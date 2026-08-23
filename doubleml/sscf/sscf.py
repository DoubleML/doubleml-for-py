"""Double machine learning for high-dimensional sample selection models
estimated with a control function (Heckman-type) correction.

The module implements the model class :class:`DoubleMLSSCF`, which follows the
object-oriented structure of the ``DoubleML`` package: only the nuisance
functions and the elements of the linear Neyman-orthogonal score are specified
here, all remaining functionality (sample splitting, estimation, inference,
bootstrap, tuning) is inherited from the abstract base class ``DoubleML``.
"""

import warnings
from typing import Optional

import numpy as np
from doubleml.data.base_data import DoubleMLData
from doubleml.double_ml import DoubleML
from doubleml.double_ml_score_mixins import LinearScoreMixin
from doubleml.utils._checks import _check_finite_predictions, _check_score
from doubleml.utils._estimation import _predict_zero_one_propensity
from doubleml.utils.propensity_score_processing import PSProcessorConfig, init_ps_processor
from scipy.stats import chi2, norm
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.utils import check_X_y

from .utils import generalized_inverse_mills_ratio


class DoubleMLSSCF(LinearScoreMixin, DoubleML):
    r"""Double machine learning for sample selection models with a control function correction.

    The model consists of an outcome (variable selection) equation and a selection
    (participation) equation

    .. math::

        Y_i &= g_0(X_i) + \varepsilon_i, \\
        D_i &= \mathbb{1}\{Z_i'\beta_0 + v_i > 0\},

    where :math:`Y_i` is observed only if :math:`D_i = 1` and :math:`Z_i = (X_i, U_i)`
    contains the outcome covariates :math:`X_i` plus additional variables :math:`U_i`
    which are excluded from the outcome equation. Under joint normality of
    :math:`(\varepsilon_i, v_i)` the selection bias in the observed subpopulation is
    :math:`\mathbb{E}[\varepsilon_i \mid X_i, Z_i, D_i = 1] = \theta_0 h_i`, with the
    inverse Mills ratio :math:`h_i = \varphi(Z_i'\beta_0) / \Phi(Z_i'\beta_0)` and
    :math:`\theta_0 = \sigma_{\varepsilon v}` the target parameter (the sample selection
    bias coefficient). The observed outcome equation reads

    .. math::

        Y_i = g_0(X_i) + \theta_0 h_i + u_i, \qquad \mathbb{E}[u_i \mid X_i] = 0 .

    Estimation is based on the linear Neyman-orthogonal score

    .. math::

        \psi(W; \theta, \eta) = \big(Y - g(X) - \theta h\big)\big(h - m(X)\big),
        \qquad \eta = (h, g, m),

    evaluated on the selected subpopulation :math:`\{D = 1\}`, with nuisance elements
    :math:`h = h(\pi(Z))`, :math:`g(X) = \mathbb{E}[Y \mid X, D = 1]` and
    :math:`m(X) = \mathbb{E}[h \mid X, D = 1]`. The inverse Mills ratio is obtained from
    the estimated participation propensity score :math:`\pi(Z) = P(D = 1 \mid Z)` via
    :math:`h = \varphi(\Phi^{-1}(\pi)) / \pi`, so that any probabilistic classifier can
    be used for the selection equation (a probit specification is the special case
    :math:`\Phi^{-1}(\pi) = Z'\beta`).

    Parameters
    ----------
    obj_dml_data : :class:`doubleml.DoubleMLData` object
        The :class:`doubleml.DoubleMLData` object providing the data. The (single)
        treatment variable ``d_cols`` has to be the **binary selection / participation
        indicator** :math:`D`, ``x_cols`` are the covariates :math:`X` of the outcome
        equation and ``z_cols`` are the additional variables :math:`U` entering only
        the selection equation (exclusion restrictions). Values of the outcome ``y``
        for :math:`D_i = 0` are never used and may be set to any finite value.

    ml_pi : classifier implementing ``fit()`` and ``predict_proba()``
        A machine learner for the participation propensity score
        :math:`\pi(Z) = P(D = 1 \mid Z)`, e.g.
        :py:class:`sklearn.linear_model.LogisticRegressionCV`.

    ml_g : regressor implementing ``fit()`` and ``predict()``
        A machine learner for the outcome regression on the selected sample
        :math:`g(X) = \mathbb{E}[Y \mid X, D = 1]`, e.g.
        :py:class:`sklearn.linear_model.LassoCV`.

    ml_m : regressor implementing ``fit()`` and ``predict()``
        A machine learner for the projection of the inverse Mills ratio on the outcome
        covariates :math:`m(X) = \mathbb{E}[h \mid X, D = 1]`, e.g.
        :py:class:`sklearn.linear_model.LassoCV`.

    n_folds : int
        Number of folds. Default is ``5``.

    n_rep : int
        Number of repetitions for the sample splitting. Default is ``1``.

    score : str
        A str (``'orthogonal'`` is the only choice) specifying the score function.
        Default is ``'orthogonal'``.

    nested_cross_fitting : bool
        Indicates whether the training folds are split in half, with the first half used
        for the selection equation and the second half for the outcome nuisances
        :math:`g` and :math:`m`. Since :math:`\hat h` is a generated regressor which is
        the target of :math:`m`, nested cross-fitting avoids own-observation bias in
        :math:`\hat m`. Default is ``True``.

    ps_processor_config : :class:`doubleml.utils.PSProcessorConfig`, optional
        Configuration for propensity score processing (clipping, calibration). The
        clipping bound is essential here since :math:`h = \varphi(\Phi^{-1}(\pi))/\pi`
        diverges for :math:`\pi \to 0` (strict overlap). Default is ``None``, which
        corresponds to clipping at ``1e-2``.

    draw_sample_splitting : bool
        Indicates whether the sample splitting should be drawn during initialization.
        The sample splitting is stratified by the selection indicator. Default is ``True``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LassoCV, LogisticRegressionCV
    >>> from doubleml_sscf import DoubleMLSSCF, make_green_silence_data
    >>> np.random.seed(3141)
    >>> dml_data = make_green_silence_data(n_obs=1000, dim_x=8, dim_z=5, n_deciles=5, theta=-0.6)
    >>> ml_pi = LogisticRegressionCV(Cs=5, max_iter=2000)
    >>> ml_g = LassoCV(n_alphas=20)
    >>> ml_m = LassoCV(n_alphas=20)
    >>> dml_obj = DoubleMLSSCF(dml_data, ml_pi, ml_g, ml_m, n_folds=3)
    >>> dml_obj.fit().summary  # doctest: +SKIP
           coef   std err         t     P>|t|     2.5 %    97.5 %
    d -0.585... 0.071...  -8.2...  2.0e-16  -0.72...  -0.44...

    Notes
    -----
    The score is defined on the selected subpopulation only; the score elements are set
    to zero for :math:`D_i = 0`. With :math:`n = \sum_i D_i` observed outcomes this
    yields the estimator

    .. math::

        \hat\theta = \frac{\sum_i D_i (Y_i - \hat g_i)(\hat h_i - \hat m_i)}
                          {\sum_i D_i \hat h_i (\hat h_i - \hat m_i)},

    and the variance estimator of the base class reduces to
    :math:`\hat\sigma_\theta^2 / n` with
    :math:`\hat\sigma^2_\theta = \hat J_0^{-2}\,\mathbb{E}_n[\psi^2 \mid D = 1]` and
    :math:`-\hat J_0 = \mathbb{E}_n[(\hat h - \hat m)^2 \mid D = 1]`, i.e. inference is
    driven by the variation in the selection propensity which is orthogonal to
    :math:`X` (the exclusion restriction).
    """

    def __init__(
        self,
        obj_dml_data,
        ml_pi,
        ml_g,
        ml_m,
        n_folds=5,
        n_rep=1,
        score="orthogonal",
        nested_cross_fitting=True,
        ps_processor_config: Optional[PSProcessorConfig] = None,
        draw_sample_splitting=True,
    ):
        # the sample splitting is drawn below, after the strata have been set
        super().__init__(obj_dml_data, n_folds, n_rep, score, draw_sample_splitting=False)

        self._external_predictions_implemented = False
        self._sensitivity_implemented = False
        self._nuisance_elements = None

        if not isinstance(nested_cross_fitting, bool):
            raise TypeError(
                "nested_cross_fitting must be True or False. "
                f"Got {nested_cross_fitting!s} of type {type(nested_cross_fitting)!s}."
            )
        self._nested_cross_fitting = nested_cross_fitting

        self._ps_processor_config, self._ps_processor = init_ps_processor(
            ps_processor_config, trimming_rule="truncate", trimming_threshold=1e-2
        )

        self._check_data(self._dml_data)
        _check_score(self.score, ["orthogonal"], allow_callable=False)

        # stratified sample splitting by the selection indicator
        self._strata = self._dml_data.d.reshape(-1, 1)
        if not isinstance(draw_sample_splitting, bool):
            raise TypeError(f"draw_sample_splitting must be True or False. Got {draw_sample_splitting!s}.")
        if draw_sample_splitting:
            self.draw_sample_splitting()

        _ = self._check_learner(ml_pi, "ml_pi", regressor=False, classifier=True)
        _ = self._check_learner(ml_g, "ml_g", regressor=True, classifier=False)
        _ = self._check_learner(ml_m, "ml_m", regressor=True, classifier=False)

        self._learner = {"ml_pi": clone(ml_pi), "ml_g": clone(ml_g), "ml_m": clone(ml_m)}
        self._predict_method = {"ml_pi": "predict_proba", "ml_g": "predict", "ml_m": "predict"}

        self._initialize_ml_nuisance_params()

    @property
    def nested_cross_fitting(self):
        """Indicates whether nested cross-fitting is used for the generated regressor."""
        return self._nested_cross_fitting

    @property
    def ps_processor_config(self):
        """Configuration for propensity score processing (clipping, calibration)."""
        return self._ps_processor_config

    @property
    def ps_processor(self):
        """Propensity score processor."""
        return self._ps_processor

    @property
    def nuisance_elements(self):
        r"""Cross-fitted nuisance elements of the last call to ``fit()``.

        Dictionary with the arrays ``'pi_hat'`` (participation propensity score),
        ``'h_hat'`` (inverse Mills ratio), ``'g_hat'`` (outcome regression) and
        ``'m_hat'`` (projection of the inverse Mills ratio), each of dimension
        ``(n_obs, n_rep)``.
        """
        return self._nuisance_elements

    @property
    def imr(self):
        r"""Estimated inverse Mills ratios :math:`\hat h_i`, averaged over repetitions."""
        if self._nuisance_elements is None:
            raise ValueError("Apply fit() before accessing imr.")
        return np.nanmean(self._nuisance_elements["h_hat"], axis=1)

    def _initialize_ml_nuisance_params(self):
        valid_learner = ["ml_pi", "ml_g", "ml_m"]
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in valid_learner}

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError(
                "For the sample selection control function model the data must be of DoubleMLData type. "
                f"{obj_dml_data!s} of type {type(obj_dml_data)!s} was passed."
            )
        if obj_dml_data.n_treat != 1:
            raise ValueError(
                "Incompatible data. To fit a DoubleMLSSCF model exactly one variable has to be specified as "
                "selection indicator via d_cols. "
                f"{obj_dml_data.n_treat!s} variables were passed."
            )
        d_values = np.unique(obj_dml_data.d)
        if not np.array_equal(np.sort(d_values), np.array([0, 1])):
            raise ValueError(
                "Incompatible data. The selection indicator (d_cols) has to be binary with values 0 and 1. "
                f"Observed values: {d_values!s}."
            )
        if obj_dml_data.z_cols is None:
            warnings.warn(
                "No exclusion restrictions were specified (z_cols is None). Identification of the sample "
                "selection bias coefficient then relies exclusively on the nonlinearity of the inverse Mills "
                "ratio in X, which is typically weak. Consider adding variables which shift participation but "
                "are excluded from the outcome equation."
            )

    def _selection_features(self):
        """Regressors Z = (X, U) of the selection equation."""
        x = self._dml_data.x
        if self._dml_data.z_cols is not None:
            return np.column_stack((x, self._dml_data.z))
        return x

    def _nuisance_est(self, smpls, n_jobs_cv, external_predictions, return_models=False):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y, ensure_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d, ensure_all_finite=False)
        z_sel = self._selection_features()

        n_obs = self._dml_data.n_obs
        pi_hat = np.full(shape=n_obs, fill_value=np.nan)
        h_hat = np.full(shape=n_obs, fill_value=np.nan)
        g_hat = np.full(shape=n_obs, fill_value=np.nan)
        m_hat = np.full(shape=n_obs, fill_value=np.nan)

        fitted_models = {}
        for learner in self.params_names:
            est_params = self._get_params(learner)
            if est_params is not None:
                fitted_models[learner] = [
                    clone(self._learner[learner]).set_params(**est_params[i_fold]) for i_fold in range(self.n_folds)
                ]
            else:
                fitted_models[learner] = [clone(self._learner[learner]) for i_fold in range(self.n_folds)]

        for i_fold in range(self.n_folds):
            train_inds, test_inds = smpls[i_fold]

            # split the training sample: first half for the selection equation, second half for the
            # outcome nuisances, since h is a generated regressor entering the target of m
            if self._nested_cross_fitting:
                train_sel, train_out = train_test_split(train_inds, test_size=0.5, random_state=42, stratify=d[train_inds])
            else:
                train_sel, train_out = train_inds, train_inds

            # (a) selection equation: participation propensity score and inverse Mills ratio
            fitted_models["ml_pi"][i_fold].fit(z_sel[train_sel, :], d[train_sel])
            pi_fold = _predict_zero_one_propensity(fitted_models["ml_pi"][i_fold], z_sel)
            pi_fold = self._ps_processor.adjust_ps(pi_fold, d, learner_name="ml_pi")
            h_fold = generalized_inverse_mills_ratio(pi_fold)

            pi_hat[test_inds] = pi_fold[test_inds]
            h_hat[test_inds] = h_fold[test_inds]

            # (b) outcome nuisances, estimated on the selected observations of the second training half
            train_out_sel = train_out[d[train_out] == 1]
            if len(train_out_sel) == 0:
                raise ValueError(
                    f"No selected observations (d == 1) in the training sample of fold {i_fold}. " "Consider reducing n_folds."
                )
            fitted_models["ml_g"][i_fold].fit(x[train_out_sel, :], y[train_out_sel])
            g_hat[test_inds] = fitted_models["ml_g"][i_fold].predict(x[test_inds, :])

            fitted_models["ml_m"][i_fold].fit(x[train_out_sel, :], h_fold[train_out_sel])
            m_hat[test_inds] = fitted_models["ml_m"][i_fold].predict(x[test_inds, :])

        _check_finite_predictions(pi_hat, self._learner["ml_pi"], "ml_pi", smpls)
        _check_finite_predictions(g_hat, self._learner["ml_g"], "ml_g", smpls)
        _check_finite_predictions(m_hat, self._learner["ml_m"], "ml_m", smpls)

        psi_a, psi_b = self._score_elements(y, d, g_hat, h_hat, m_hat)
        psi_elements = {"psi_a": psi_a, "psi_b": psi_b}

        # targets of g and m are only defined on the selected subpopulation
        y_target = np.where(d == 1, y, np.nan)
        h_target = np.where(d == 1, h_hat, np.nan)

        preds = {
            "predictions": {"ml_pi": pi_hat, "ml_g": g_hat, "ml_m": m_hat},
            "targets": {"ml_pi": d.astype(float), "ml_g": y_target, "ml_m": h_target},
            "models": {
                "ml_pi": fitted_models["ml_pi"] if return_models else None,
                "ml_g": fitted_models["ml_g"] if return_models else None,
                "ml_m": fitted_models["ml_m"] if return_models else None,
            },
        }
        # model specific elements, stored for the score test and for the bias correction
        if self._nuisance_elements is None:
            self._nuisance_elements = {
                key: np.full((n_obs, self.n_rep), np.nan) for key in ["pi_hat", "h_hat", "g_hat", "m_hat"]
            }
        i_rep = 0 if self._i_rep is None else self._i_rep
        for key, value in zip(["pi_hat", "h_hat", "g_hat", "m_hat"], [pi_hat, h_hat, g_hat, m_hat]):
            self._nuisance_elements[key][:, i_rep] = value

        return psi_elements, preds

    def _score_elements(self, y, d, g_hat, h_hat, m_hat):
        """Elements of the linear score psi = psi_a * theta + psi_b, zero outside {D = 1}."""
        h_tilde = h_hat - m_hat
        psi_a = -d * h_hat * h_tilde
        psi_b = d * (y - g_hat) * h_tilde
        return psi_a, psi_b

    def selection_bias_test(self, level=0.05):
        r"""Doubly robust orthogonal score test for the absence of sample selection bias.

        The test statistic evaluates the Neyman-orthogonal score at :math:`\theta = 0`,

        .. math::

            S_n = \frac{\big(\sum_i D_i \psi(W_i; 0, \hat\eta)\big)^2}
                       {\sum_i D_i \big(\psi(W_i; 0, \hat\eta) - \bar\psi_n\big)^2}
            \xrightarrow{d} \chi^2_1 \quad \text{under } H_0: \theta_0 = 0 .

        Parameters
        ----------
        level : float
            Significance level of the test. Default is ``0.05``.

        Returns
        -------
        res : dict
            Dictionary with the test statistic (``'statistic'``), the p-value
            (``'p_value'``), the critical value (``'critical_value'``), the test decision
            (``'reject'``) and the number of selected observations (``'n_selected'``).
        """
        if self._framework is None:
            raise ValueError("Apply fit() before selection_bias_test().")
        if (level <= 0) or (level >= 1):
            raise ValueError(f"The significance level must be in (0, 1). {level!s} was passed.")

        d = self._dml_data.d
        selected = d == 1
        n_selected = int(np.sum(selected))

        stats = np.full(self.n_rep, np.nan)
        for i_rep in range(self.n_rep):
            psi_null = self.psi_elements["psi_b"][:, i_rep, 0][selected]
            score_mean = np.mean(psi_null)
            score_var = np.mean(np.square(psi_null - score_mean))
            stats[i_rep] = n_selected * score_mean**2 / score_var

        # aggregation over repeated sample splitting via the median (as for the coefficients)
        statistic = float(np.median(stats))
        p_value = float(chi2.sf(statistic, df=1))
        critical_value = float(chi2.ppf(1 - level, df=1))

        return {
            "statistic": statistic,
            "p_value": p_value,
            "critical_value": critical_value,
            "reject": bool(statistic > critical_value),
            "n_selected": n_selected,
        }

    def bias_quantification(self):
        r"""Quantification of the sample selection bias implied by the fitted model.

        Returns the estimated bias term :math:`\hat\theta \hat h_i` for the selected
        subpopulation, the bias-corrected outcomes :math:`Y_i - \hat\theta \hat h_i`
        and summary statistics. The average bias
        :math:`\hat\theta\, \mathbb{E}_n[\hat h_i \mid D_i = 1]` measures by how much
        outcomes of the selected (disclosing) units are shifted relative to the
        structural outcome equation; the counterfactual bias for the non-selected units
        is evaluated at :math:`\hat h_i` implied by their own participation propensity.

        Returns
        -------
        res : dict
            Dictionary with the estimated inverse Mills ratios (``'imr'``), the bias
            term (``'bias'``), the corrected outcomes on the selected sample
            (``'y_corrected'``) and summary statistics (``'summary'``).
        """
        if self._framework is None:
            raise ValueError("Apply fit() before bias_quantification().")

        d = self._dml_data.d
        y = self._dml_data.y
        theta = float(self.coef[0])
        h_hat = self.imr

        bias = theta * h_hat
        y_corrected = np.where(d == 1, y - bias, np.nan)

        summary = {
            "theta": theta,
            "se": float(self.se[0]),
            "mean_imr_selected": float(np.mean(h_hat[d == 1])),
            "mean_imr_non_selected": float(np.mean(h_hat[d == 0])) if np.any(d == 0) else np.nan,
            "mean_bias_selected": float(np.mean(bias[d == 1])),
            "mean_bias_non_selected": float(np.mean(bias[d == 0])) if np.any(d == 0) else np.nan,
            "selection_rate": float(np.mean(d)),
        }
        return {"imr": h_hat, "bias": bias, "y_corrected": y_corrected, "summary": summary}

    def _nuisance_tuning(
        self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search
    ):
        from doubleml.utils._estimation import _dml_tune

        x, y = check_X_y(self._dml_data.x, self._dml_data.y, ensure_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d, ensure_all_finite=False)
        z_sel = self._selection_features()

        if scoring_methods is None:
            scoring_methods = {"ml_pi": None, "ml_g": None, "ml_m": None}

        train_inds = [train_index for (train_index, _) in smpls]
        train_inds_sel = [train_index[d[train_index] == 1] for (train_index, _) in smpls]

        pi_tune_res = _dml_tune(
            d,
            z_sel,
            train_inds,
            self._learner["ml_pi"],
            param_grids["ml_pi"],
            scoring_methods["ml_pi"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )
        g_tune_res = _dml_tune(
            y,
            x,
            train_inds_sel,
            self._learner["ml_g"],
            param_grids["ml_g"],
            scoring_methods["ml_g"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )

        # the target of m is the generated regressor; for tuning it is approximated by a
        # full-sample fit of the selection equation
        pi_full = np.full(shape=self._dml_data.n_obs, fill_value=np.nan)
        for i_fold, (train_index, test_index) in enumerate(smpls):
            ml_pi_temp = clone(self._learner["ml_pi"])
            ml_pi_temp.set_params(**pi_tune_res[i_fold].best_params_)
            ml_pi_temp.fit(z_sel[train_index, :], d[train_index])
            pi_full[test_index] = _predict_zero_one_propensity(ml_pi_temp, z_sel)[test_index]
        pi_full = self._ps_processor.adjust_ps(pi_full, d, learner_name="ml_pi")
        h_full = generalized_inverse_mills_ratio(pi_full)

        m_tune_res = _dml_tune(
            h_full,
            x,
            train_inds_sel,
            self._learner["ml_m"],
            param_grids["ml_m"],
            scoring_methods["ml_m"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )

        params = {
            "ml_pi": [res.best_params_ for res in pi_tune_res],
            "ml_g": [res.best_params_ for res in g_tune_res],
            "ml_m": [res.best_params_ for res in m_tune_res],
        }
        tune_res = {"pi_tune": pi_tune_res, "g_tune": g_tune_res, "m_tune": m_tune_res}

        return {"params": params, "tune_res": tune_res}

    def _nuisance_tuning_optuna(self, optuna_params, scoring_methods, cv, optuna_settings):
        raise NotImplementedError("Optuna tuning is not implemented for DoubleMLSSCF.")

    def _sensitivity_element_est(self, preds):
        pass


def imr_from_index(index):
    """Inverse Mills ratio evaluated at a linear index, :math:`\\varphi(z)/\\Phi(z)`."""
    index = np.asarray(index, dtype=float)
    return norm.pdf(index) / np.clip(norm.cdf(index), 1e-12, None)
