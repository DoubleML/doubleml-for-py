import inspect
import warnings

import numpy as np
import scipy
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target

from doubleml.double_ml import DoubleML
from doubleml.double_ml_score_mixins import NonLinearScoreMixin
from doubleml.utils._checks import _check_finite_predictions, _check_is_propensity, _check_score
from doubleml.utils._estimation import (
    _dml_cv_predict,
    _dml_tune,
    _double_dml_cv_predict,
)
from doubleml.utils._tune_optuna import _dml_tune_optuna


class DoubleMLLPLR(NonLinearScoreMixin, DoubleML):
    """Double machine learning for partially logistic models (binary outcomes).

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.
        The outcome variable y must be binary with values {0, 1}.

    ml_M : estimator implementing ``fit()`` and ``predict_proba()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance function
        :math:`M_0(D, X) = P[Y = 1 | D, X]`.

    ml_t : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the auxiliary regression
        used to predict log-odds :math:`t_0(X) = E[W | X]` where :math:`W = \\text{logit}(M_0(D, X))`.

    ml_m : estimator implementing ``fit()`` and ``predict()`` or ``predict_proba()``
        A machine learner for the nuisance function :math:`m_0(X) = E[D | X]`.
        For binary treatments, a classifier implementing ``fit()`` and ``predict_proba()`` is expected
        (e.g. :py:class:`sklearn.ensemble.RandomForestClassifier`).
        For continuous treatments, a regressor implementing ``fit()`` and ``predict()`` is expected
        (e.g. :py:class:`sklearn.ensemble.RandomForestRegressor`).

    ml_a : estimator implementing ``fit()`` and ``predict()`` or ``predict_proba()``, optional
        Optional alternative learner for :math:`E[D | X]`. If not provided, a clone of ``ml_m`` is used.
        Must support the same prediction interface as ``ml_m``.
        Default is ``None``.

    n_folds : int
        Number of outer cross-fitting folds.
        Default is ``5``.

    n_folds_inner : int
        Number of inner folds for nested resampling used internally.
        Default is ``5``.

    n_rep : int
        Number of repetitions for the sample splitting.
        Default is ``1``.

    score : str
        A str (``'nuisance_space'`` or ``'instrument'``) specifying the score function.
        ``'nuisance_space'`` estimates m on subsamples with y=0;
        ``'instrument'`` uses an instrument-type score.
        Default is ``'nuisance_space'``.

    draw_sample_splitting : bool
        Indicates whether the sample splitting should be drawn during initialization of the object.
        Default is ``True``.

    error_on_convergence_failure : bool
        If ``True``, raise an error on convergence failure of score.
        Default is ``False``.

    Examples
    --------
    >>> import numpy as np
    >>> import doubleml as dml
    >>> from doubleml.plm.datasets import make_lplr_LZZ2020
    >>> from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    >>> from sklearn.base import clone
    >>> np.random.seed(42)
    >>> ml_t = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
    >>> ml_m = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
    >>> ml_M = RandomForestClassifier(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
    >>> obj_dml_data = make_lplr_LZZ2020(alpha=0.5, n_obs=500, dim_x=20)
    >>> dml_lplr_obj = dml.DoubleMLLPLR(obj_dml_data, ml_M, ml_t, ml_m)
    >>> dml_lplr_obj.fit().summary  # doctest: +SKIP
           coef   std err         t     P>|t|     2.5 %    97.5 %
    d  0.661166  0.172672  3.829038  0.000129  0.322736  0.999596

    Notes
    -----
    **Partially logistic regression (PLR)** models take the form

    .. math::

        Y =  \\text{expit} ( D \\theta_0 + r_0(X))

    where :math:`Y` is the outcome variable and :math:`D` is the policy variable of interest.
    The (potentially) high-dimensional vector :math:`X = (X_1, \\ldots, X_p)` consists of other confounding covariates.
    """

    def __init__(
        self,
        obj_dml_data,
        ml_M,
        ml_t,
        ml_m,
        ml_a=None,
        n_folds=5,
        n_folds_inner=5,
        n_rep=1,
        score="nuisance_space",
        draw_sample_splitting=True,
        error_on_convergence_failure=False,
    ):
        self.n_folds_inner = n_folds_inner
        super().__init__(obj_dml_data, n_folds, n_rep, score, draw_sample_splitting, double_sample_splitting=True)

        self._error_on_convergence_failure = error_on_convergence_failure

        self._check_data(self._dml_data)
        valid_scores = ["nuisance_space", "instrument"]
        _check_score(self.score, valid_scores, allow_callable=False)

        _ = self._check_learner(ml_t, "ml_t", regressor=True, classifier=False)
        _ = self._check_learner(ml_M, "ml_M", regressor=False, classifier=True)

        ml_m_is_classifier = self._check_learner(ml_m, "ml_m", regressor=True, classifier=True)
        self._learner = {"ml_m": ml_m, "ml_t": ml_t, "ml_M": ml_M}

        if ml_a is not None:
            ml_a_is_classifier = self._check_learner(ml_a, "ml_a", regressor=True, classifier=True)
            self._learner["ml_a"] = ml_a
            self._ml_a_provided = True
        else:
            self._learner["ml_a"] = clone(ml_m)
            ml_a_is_classifier = ml_m_is_classifier
            self._ml_a_provided = False

        self._predict_method = {"ml_t": "predict", "ml_M": "predict_proba"}

        if ml_m_is_classifier:
            if self._dml_data.binary_treats.all():
                self._predict_method["ml_m"] = "predict_proba"
            else:
                raise ValueError(
                    f"The ml_m learner {str(ml_m)} was identified as classifier "
                    "but at least one treatment variable is not binary with values 0 and 1."
                )
        else:
            if self._dml_data.binary_treats.any():
                warnings.warn(
                    f"The ml_m learner {str(ml_m)} was identified as regressor "
                    "but at least one treatment variable is binary with values 0 and 1."
                )
            self._predict_method["ml_m"] = "predict"

        if ml_a_is_classifier:
            if self._dml_data.binary_treats.all():
                self._predict_method["ml_a"] = "predict_proba"
            else:
                raise ValueError(
                    f"The ml_a learner {str(ml_a)} was identified as classifier "
                    "but at least one treatment variable is not binary with values 0 and 1."
                )
        else:
            if self._dml_data.binary_treats.any():
                warnings.warn(
                    f"The ml_a learner {str(ml_a)} was identified as regressor but at least one treatment variable is "
                    f"binary with values 0 and 1."
                )
            self._predict_method["ml_a"] = "predict"

        if score == "instrument":
            sig = inspect.signature(self.learner["ml_a"].fit)
            if "sample_weight" not in sig.parameters:
                raise ValueError('Learner "ml_a" who supports sample_weight is required for score type "instrument"')

        self._initialize_ml_nuisance_params()
        self._external_predictions_implemented = True
        self._sensitivity_implemented = False

    def _initialize_ml_nuisance_params(self):
        inner_M_names = [f"ml_M_inner_{i}" for i in range(self.n_folds)]
        inner_a_names = [f"ml_a_inner_{i}" for i in range(self.n_folds)]
        params_names = ["ml_m", "ml_a", "ml_t", "ml_M"] + inner_M_names + inner_a_names
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in params_names}

    def _check_data(self, obj_dml_data):
        if not np.array_equal(np.unique(obj_dml_data.y), [0, 1]):
            raise TypeError("The outcome variable y must be binary with values 0 and 1.")

    def _nuisance_est(self, smpls, n_jobs_cv, external_predictions, return_models=False):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y, ensure_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d, ensure_all_finite=False)
        x_d_concat = np.hstack((d.reshape(-1, 1), x))
        m_external = external_predictions["ml_m"] is not None
        M_external = external_predictions["ml_M"] is not None
        t_external = external_predictions["ml_t"] is not None
        a_external = external_predictions["ml_a"] is not None

        if M_external:
            # expect per-inner-fold keys ml_M_inner_i
            missing = [
                i
                for i in range(self.n_folds_inner)
                if f"ml_M_inner_{i}" not in external_predictions.keys() or external_predictions[f"ml_M_inner_{i}"] is None
            ]
            if len(missing) > 0:
                raise ValueError(
                    "When providing external predictions for ml_M, also inner predictions for all inner folds "
                    f"have to be provided (missing: {', '.join([str(i) for i in missing])})."
                )
            M_hat_inner = [external_predictions[f"ml_M_inner_{i}"] for i in range(self.n_folds_inner)]
            M_hat = {
                "preds": external_predictions["ml_M"],
                "preds_inner": M_hat_inner,
                "targets": self._dml_data.y,
                "models": None,
            }
        else:
            M_hat = _double_dml_cv_predict(
                self._learner["ml_M"],
                "ml_M",
                x_d_concat,
                y,
                smpls=smpls,
                smpls_inner=self._DoubleML__smpls__inner,
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_M"),
                method=self._predict_method["ml_M"],
            )

        # nuisance m
        if m_external:
            m_hat = {"preds": external_predictions["ml_m"], "targets": self._dml_data.d, "models": None}
        else:
            if self.score == "instrument":
                weights = M_hat["preds"] * (1 - M_hat["preds"])
                filtered_smpls = smpls
            elif self.score == "nuisance_space":
                filtered_smpls = []
                for train, test in smpls:
                    train_filtered = train[y[train] == 0]
                    filtered_smpls.append((train_filtered, test))
                weights = None

            m_hat = _dml_cv_predict(
                self._learner["ml_m"],
                x,
                d,
                smpls=filtered_smpls,
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_m"),
                method=self._predict_method["ml_m"],
                return_models=return_models,
                sample_weights=weights,
            )

            _check_finite_predictions(m_hat["preds"], self._learner["ml_m"], "ml_m", smpls)

        if self._check_learner(self._learner["ml_m"], "ml_m", regressor=True, classifier=True):
            _check_is_propensity(m_hat["preds"], self._learner["ml_m"], "ml_m", smpls, eps=1e-12)

        if self._dml_data.binary_treats[self._dml_data.d_cols[self._i_treat]]:
            binary_preds = type_of_target(m_hat["preds"]) == "binary"
            zero_one_preds = np.all((np.power(m_hat["preds"], 2) - m_hat["preds"]) == 0)
            if binary_preds & zero_one_preds:
                raise ValueError(
                    f"For the binary treatment variable {self._dml_data.d_cols[self._i_treat]}, "
                    f"predictions obtained with the ml_m learner {str(self._learner['ml_m'])} are also "
                    "observed to be binary with values 0 and 1. Make sure that for classifiers "
                    "probabilities and not labels are predicted."
                )

        if a_external:
            # expect per-inner-fold keys ml_a_inner_i
            missing = [
                i
                for i in range(self.n_folds_inner)
                if f"ml_a_inner_{i}" not in external_predictions.keys() or external_predictions[f"ml_a_inner_{i}"] is None
            ]
            if len(missing) > 0:
                raise ValueError(
                    "When providing external predictions for ml_a, also inner predictions for all inner folds "
                    f"have to be provided (missing: {', '.join([str(i) for i in missing])})."
                )
            a_hat_inner = [external_predictions[f"ml_a_inner_{i}"] for i in range(self.n_folds_inner)]
            a_hat = {
                "preds": external_predictions["ml_a"],
                "preds_inner": a_hat_inner,
                "targets": self._dml_data.d,
                "models": None,
            }
        else:
            a_hat = _double_dml_cv_predict(
                self._learner["ml_a"],
                "ml_a",
                x,
                d,
                smpls=smpls,
                smpls_inner=self._DoubleML__smpls__inner,
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_a"),
                method=self._predict_method["ml_a"],
            )

        W_inner = []
        beta = np.zeros(d.shape, dtype=float)

        for i, (train, test) in enumerate(smpls):
            M_iteration = M_hat["preds_inner"][i][train]
            M_iteration = np.clip(M_iteration, 1e-8, 1 - 1e-8)
            w = scipy.special.logit(M_iteration)
            W_inner.append(w)
            d_tilde = (d - a_hat["preds_inner"][i])[train]
            beta[test] = np.sum(d_tilde * w) / np.sum(d_tilde**2)

        # Use preliminary beta estimates as starting value for root finding
        self._coef_start_val = np.average(beta)

        # nuisance t
        if t_external:
            t_hat = {"preds": external_predictions["ml_t"], "targets": None, "models": None}
        else:
            t_hat = _dml_cv_predict(
                self._learner["ml_t"],
                x,
                W_inner,
                smpls=smpls,
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_t"),
                method=self._predict_method["ml_t"],
                return_models=return_models,
            )
            _check_finite_predictions(t_hat["preds"], self._learner["ml_t"], "ml_t", smpls)

        r_hat = {}
        r_hat["preds"] = t_hat["preds"] - beta * a_hat["preds"]

        psi_elements = self._score_elements(y, d, r_hat["preds"], m_hat["preds"])

        preds = {
            "predictions": {
                "ml_r": r_hat["preds"],
                "ml_m": m_hat["preds"],
                "ml_a": a_hat["preds"],
                "ml_t": t_hat["preds"],
                "ml_M": M_hat["preds"],
                # store inner predictions as separate keys per inner fold
                # ml_M inner
                **{f"ml_M_inner_{i}": M_hat["preds_inner"][i] for i in range(len(M_hat["preds_inner"]))},
                # ml_a inner
                **{f"ml_a_inner_{i}": a_hat["preds_inner"][i] for i in range(len(a_hat["preds_inner"]))},
            },
            "targets": {
                "ml_r": None,
                "ml_m": m_hat["targets"],
                "ml_a": a_hat["targets"],
                "ml_t": t_hat["targets"],
                "ml_M": M_hat["targets"],
                # store inner targets as separate keys per inner fold (None if external)
                **(
                    {
                        f"ml_M_inner_{i}": (
                            M_hat.get("targets_inner")[i]
                            if M_hat.get("targets_inner") is not None and i < len(M_hat["targets_inner"])
                            else None
                        )
                        for i in range(len(M_hat.get("preds_inner", [])))
                    }
                ),
                **(
                    {
                        f"ml_a_inner_{i}": (
                            a_hat.get("targets_inner")[i]
                            if a_hat.get("targets_inner") is not None and i < len(a_hat["targets_inner"])
                            else None
                        )
                        for i in range(len(a_hat.get("preds_inner", [])))
                    }
                ),
            },
            "models": {
                "ml_r": None,
                "ml_m": m_hat["models"],
                "ml_a": a_hat["models"],
                "ml_t": t_hat["models"],
                "ml_M": M_hat["models"],
            },
        }

        return psi_elements, preds

    def _score_elements(self, y, d, r_hat, m_hat):
        # compute residual
        d_tilde = d - m_hat
        psi_hat = scipy.special.expit(-r_hat)
        score_const = d_tilde * (1 - y) * np.exp(r_hat)
        psi_elements = {
            "y": y,
            "d": d,
            "d_tilde": d_tilde,
            "r_hat": r_hat,
            "m_hat": m_hat,
            "psi_hat": psi_hat,
            "score_const": score_const,
        }

        return psi_elements

    @property
    def _score_element_names(self):
        return ["y", "d", "d_tilde", "r_hat", "m_hat", "psi_hat", "score_const"]

    def _sensitivity_element_est(self, preds):
        raise NotImplementedError()

    def _nuisance_tuning(
        self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search
    ):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y, ensure_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d, ensure_all_finite=False)
        x_d_concat = np.hstack((d.reshape(-1, 1), x))

        if scoring_methods is None:
            scoring_methods = {"ml_m": None, "ml_M": None, "ml_a": None, "ml_t": None}

        train_inds = [train_index for (train_index, _) in smpls]
        M_tune_res = _dml_tune(
            y,
            x_d_concat,
            train_inds,
            self._learner["ml_M"],
            param_grids["ml_M"],
            scoring_methods["ml_M"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )

        filtered_train_inds = []
        if self.score == "nuisance_space":
            for train, _ in smpls:
                train_filtered = train[y[train] == 0]
                filtered_train_inds.append(train_filtered)
        elif self.score == "instrument":
            filtered_train_inds = train_inds

        m_tune_res = _dml_tune(
            d,
            x,
            filtered_train_inds,
            self._learner["ml_m"],
            param_grids["ml_m"],
            scoring_methods["ml_m"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )

        a_tune_res = _dml_tune(
            d,
            x,
            train_inds,
            self._learner["ml_a"],
            param_grids["ml_a"],
            scoring_methods["ml_a"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )

        M_best_params = [xx.best_params_ for xx in M_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]
        a_best_params = [xx.best_params_ for xx in a_tune_res]

        # Create targets for tuning ml_t
        # Unlike for inference in _nuisance_est, we do not use the double cross-fitting here and use a single model for
        # predicting M_hat
        # This presents a small risk of bias in the targets, but enables tuning without tune_on_folds=True

        M_hat = np.full_like(y, np.nan)
        for idx, (train_index, _) in enumerate(smpls):
            M_hat[train_index] = M_tune_res[idx].predict_proba(x_d_concat[train_index, :])[:, 1]

        M_hat = np.clip(M_hat, 1e-8, 1 - 1e-8)
        W_hat = scipy.special.logit(M_hat)

        t_tune_res = _dml_tune(
            W_hat,
            x,
            train_inds,
            self._learner["ml_t"],
            param_grids["ml_t"],
            scoring_methods["ml_t"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )
        t_best_params = [xx.best_params_ for xx in t_tune_res]

        # Update params and tune_res to include ml_a and ml_t
        params = {"ml_M": M_best_params, "ml_m": m_best_params, "ml_a": a_best_params, "ml_t": t_best_params}
        tune_res = {"M_tune": M_tune_res, "m_tune": m_tune_res, "a_tune": a_tune_res, "t_tune": t_tune_res}

        res = {"params": params, "tune_res": tune_res}

        return res

    def _compute_score(self, psi_elements, coef):
        if self.score == "nuisance_space":
            score_1 = psi_elements["y"] * np.exp(-coef * psi_elements["d"]) * psi_elements["d_tilde"]
            score = psi_elements["psi_hat"] * (score_1 - psi_elements["score_const"])
        elif self.score == "instrument":
            score = (psi_elements["y"] - scipy.special.expit(coef * psi_elements["d"] + psi_elements["r_hat"])) * psi_elements[
                "d_tilde"
            ]

        return score

    def _compute_score_deriv(self, psi_elements, coef, inds=None):
        if self.score == "nuisance_space":
            deriv_1 = -psi_elements["y"] * np.exp(-coef * psi_elements["d"]) * psi_elements["d"]
            deriv = psi_elements["psi_hat"] * psi_elements["d_tilde"] * deriv_1
        elif self.score == "instrument":
            expit = scipy.special.expit(coef * psi_elements["d"] + psi_elements["r_hat"])
            deriv = -psi_elements["d"] * expit * (1 - expit) * psi_elements["d_tilde"]

        return deriv

    def _nuisance_tuning_optuna(
        self,
        optuna_params,
        scoring_methods,
        cv,
        optuna_settings,
    ):
        """
        Optuna-based hyperparameter tuning for LPLR nuisance models.

        Performs tuning once on the whole dataset using cross-validation,
        returning the same optimal parameters for all folds.
        """
        x, y = check_X_y(self._dml_data.x, self._dml_data.y, ensure_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d, ensure_all_finite=False)
        x_d_concat = np.hstack((d.reshape(-1, 1), x))

        if scoring_methods is None:
            scoring_methods = {"ml_m": None, "ml_M": None, "ml_a": None, "ml_t": None}

        M_tune_res = _dml_tune_optuna(
            y,
            x_d_concat,
            self._learner["ml_M"],
            optuna_params["ml_M"],
            scoring_methods["ml_M"],
            cv,
            optuna_settings,
            learner_name="ml_M",
            params_name="ml_M",
        )

        if self.score == "nuisance_space":
            mask_y0 = y == 0
            outcome_ml_m = d[mask_y0]
            features_ml_m = x[mask_y0, :]
        elif self.score == "instrument":
            outcome_ml_m = d
            features_ml_m = x

        m_tune_res = _dml_tune_optuna(
            outcome_ml_m,
            features_ml_m,
            self._learner["ml_m"],
            optuna_params["ml_m"],
            scoring_methods["ml_m"],
            cv,
            optuna_settings,
            learner_name="ml_m",
            params_name="ml_m",
        )

        a_tune_res = _dml_tune_optuna(
            d,
            x,
            self._learner["ml_a"],
            optuna_params["ml_a"],
            scoring_methods["ml_a"],
            cv,
            optuna_settings,
            learner_name="ml_a",
            params_name="ml_a",
        )

        # Create targets for tuning ml_t
        # Unlike for inference in _nuisance_est, we do not use the double cross-fitting here and use a single model for
        # predicting M_hat
        # This presents a small risk of bias in the targets, but enables tuning without tune_on_folds=True

        M_hat = cross_val_predict(
            estimator=clone(M_tune_res.best_estimator),
            X=x_d_concat,
            y=y,
            cv=cv,
            method="predict_proba",
        )[:, 1]
        M_hat = np.clip(M_hat, 1e-8, 1 - 1e-8)
        W_hat = scipy.special.logit(M_hat)

        t_tune_res = _dml_tune_optuna(
            W_hat,
            x,
            self._learner["ml_t"],
            optuna_params["ml_t"],
            scoring_methods["ml_t"],
            cv,
            optuna_settings,
            learner_name="ml_t",
            params_name="ml_t",
        )

        results = {
            "ml_M": M_tune_res,
            "ml_m": m_tune_res,
            "ml_a": a_tune_res,
            "ml_t": t_tune_res,
        }
        return results
