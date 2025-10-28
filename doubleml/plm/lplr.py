import inspect

import numpy as np
import scipy
from sklearn.base import clone
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target

from doubleml import DoubleMLData
from doubleml.double_ml import DoubleML
from doubleml.double_ml_score_mixins import NonLinearScoreMixin
from doubleml.utils._checks import _check_finite_predictions, _check_is_propensity, _check_score
from doubleml.utils._estimation import (
    _dml_cv_predict,
    _dml_tune,
)
from doubleml.utils.resampling import DoubleMLDoubleResampling


class DoubleMLLPLR(NonLinearScoreMixin, DoubleML):
    """Double machine learning for partially logistic models (binary outcomes)

    Parameters
    ----------
    obj_dml_data : DoubleMLData
        The DoubleMLData object providing the data and variable specification.
        The outcome variable y must be binary with values {0, 1}.
    ml_M : estimator
        Classifier for M_0(D, X) = P[Y = 1 | D, X]. Must implement fit() and predict_proba().
    ml_t : estimator
        Regressor for the auxiliary regression used to predict log-odds. Must implement fit() and predict().
    ml_m : estimator
        Learner for m_0(X) = E[D | X]. For binary treatments a classifier with predict_proba() is expected;
        for continuous treatments a regressor with predict() is expected.
    ml_a : estimator, optional
        Optional alternative learner for E[D | X]. If not provided, a clone of ml_m is used.
        Must support the same prediction interface as ml_m.
    n_folds : int, default=5
        Number of outer cross-fitting folds.
    n_folds_inner : int, default=5
        Number of inner folds for nested resampling used internally.
    n_rep : int, default=1
        Number of repetitions for sample splitting.
    score : {'nuisance_space', 'instrument'} or callable, default='nuisance_space'
        Score to use. 'nuisance_space' estimates m on subsamples with y=0; 'instrument' uses an instrument-type score.
    draw_sample_splitting : bool, default=True
        Whether to draw sample splitting during initialization.
    error_on_convergence_failure : bool, default=False
        If True, raise an error on convergence failure of score.

    Examples
    --------
    >>> import numpy as np
    >>> import doubleml as dml
    >>> from doubleml.plm.datasets import make_lplr_LZZ2020
    >>> from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    >>> from sklearn.base import clone
    >>> np.random.seed(3141)
    >>> ml_t = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
    >>> ml_m = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
    >>> ml_M = RandomForestClassifier(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
    >>> obj_dml_data = make_lplr_LZZ2020(alpha=0.5, n_obs=500, dim_x=20)
    >>> dml_lplr_obj = dml.DoubleMLPLR(obj_dml_data, ml_M, ml_t, ml_m)
    >>> dml_lplr_obj.fit().summary
           coef   std err          t         P>|t|     2.5 %    97.5 %
    d  0.480691  0.040533  11.859129  1.929729e-32  0.401247  0.560135

    Notes
    -----
    **Partially logistic regression (PLR)** models take the form

    .. math::

        Y =  \\text{expit} ( D \\theta_0 + r_0(X))

    where :math:`Y` is the outcome variable and :math:`D` is the policy variable of interest.
    The high-dimensional vector :math:`X = (X_1, \\ldots, X_p)` consists of other confounding covariates.
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
        super().__init__(obj_dml_data, n_folds, n_rep, score, draw_sample_splitting)

        # Ensure outcome only contains 0 and 1 (validate early in constructor)
        if not np.array_equal(np.unique(obj_dml_data.y), [0, 1]):
            raise TypeError("The outcome variable y must be binary with values 0 and 1.")

        self._error_on_convergence_failure = error_on_convergence_failure
        self._coef_bounds = (-1e-2, 1e2)
        self._coef_start_val = 1.0

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
            self._predict_method["ml_a"] = "predict"

        if score == "instrument":
            sig = inspect.signature(self.learner["ml_a"].fit)
            if "sample_weight" not in sig.parameters:
                raise ValueError('Learner "ml_a" who supports sample_weight is required for score type "instrument"')

        self._initialize_ml_nuisance_params()
        self._external_predictions_implemented = True

    def _initialize_ml_nuisance_params(self):
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in self._learner}

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError(
                f"The data must be of DoubleMLData type. {str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed."
            )
        if not np.array_equal(np.unique(obj_dml_data.y), [0, 1]):
            raise TypeError("The outcome variable y must be binary with values 0 and 1.")
        return

    def _double_dml_cv_predict(
        self,
        estimator,
        estimator_name,
        x,
        y,
        smpls=None,
        smpls_inner=None,
        n_jobs=None,
        est_params=None,
        method="predict",
        sample_weights=None,
    ):
        res = {}
        res["preds"] = np.zeros(y.shape, dtype=float)
        res["preds_inner"] = []
        res["models"] = []
        for smpls_single_split, smpls_double_split in zip(smpls, smpls_inner):
            res_inner = _dml_cv_predict(
                estimator,
                x,
                y,
                smpls=smpls_double_split,
                n_jobs=n_jobs,
                est_params=est_params,
                method=method,
                return_models=True,
                smpls_is_partition=True,
                sample_weights=sample_weights,
            )
            _check_finite_predictions(res_inner["preds"], estimator, estimator_name, smpls_double_split)

            res["preds_inner"].append(res_inner["preds"])
            for model in res_inner["models"]:
                res["models"].append(model)
                if method == "predict_proba":
                    res["preds"][smpls_single_split[1]] += model.predict_proba(x[smpls_single_split[1]])[:, 1]
                else:
                    res["preds"][smpls_single_split[1]] += model.predict(x[smpls_single_split[1]])
        res["preds"] /= len(smpls)
        res["targets"] = np.copy(y)
        return res

    def _nuisance_est(self, smpls, n_jobs_cv, external_predictions, return_models=False):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y, force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d, force_all_finite=False)
        x_d_concat = np.hstack((d.reshape(-1, 1), x))
        m_external = external_predictions["ml_m"] is not None
        M_external = external_predictions["ml_M"] is not None
        t_external = external_predictions["ml_t"] is not None
        if "ml_a" in self._learner:
            a_external = external_predictions["ml_a"] is not None
        else:
            a_external = False

        if M_external:
            M_hat = {"preds": external_predictions["ml_M"], "targets": None, "models": None}
        else:
            M_hat = self._double_dml_cv_predict(
                self._learner["ml_M"],
                "ml_M",
                x_d_concat,
                y,
                smpls=smpls,
                smpls_inner=self.__smpls__inner,
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_M"),
                method=self._predict_method["ml_M"],
            )

        # nuisance m
        if m_external:
            m_hat = {"preds": external_predictions["ml_m"], "targets": None, "models": None}
        else:
            if self.score == "instrument":
                weights = []
                for i, (train, test) in enumerate(smpls):
                    weights.append(M_hat["preds_inner"][i][train] * (1 - M_hat["preds_inner"][i][train]))
                m_hat = _dml_cv_predict(
                    self._learner["ml_m"],
                    x,
                    d,
                    smpls=smpls,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_m"),
                    method=self._predict_method["ml_m"],
                    return_models=return_models,
                    sample_weights=weights,
                )

            elif self.score == "nuisance_space":
                filtered_smpls = []
                for train, test in smpls:
                    train_filtered = train[y[train] == 0]
                    filtered_smpls.append((train_filtered, test))
                m_hat = _dml_cv_predict(
                    self._learner["ml_m"],
                    x,
                    d,
                    smpls=filtered_smpls,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_m"),
                    method=self._predict_method["ml_m"],
                    return_models=return_models,
                )
            else:
                raise NotImplementedError
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
            a_hat = {"preds": external_predictions["ml_a"], "targets": None, "models": None}
        else:
            a_hat = self._double_dml_cv_predict(
                self._learner["ml_a"],
                "ml_a",
                x,
                d,
                smpls=smpls,
                smpls_inner=self.__smpls__inner,
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
            },
            "targets": {
                "ml_r": None,
                "ml_m": m_hat["targets"],
                "ml_a": a_hat["targets"],
                "ml_t": t_hat["targets"],
                "ml_M": M_hat["targets"],
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
        pass

    def _nuisance_tuning(
        self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search
    ):
        # TODO: test
        x, y = check_X_y(self._dml_data.x, self._dml_data.y, force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d, force_all_finite=False)
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
            for train, test in smpls:
                train_filtered = train[y[train] == 0]
                filtered_train_inds.append(train_filtered)
        elif self.score == "instrument":
            filtered_train_inds = train_inds
        else:
            raise NotImplementedError
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
        M_hat = self._double_dml_cv_predict(
            self._learner["ml_M"],
            "ml_M",
            x_d_concat,
            y,
            smpls=smpls,
            smpls_inner=self.__smpls__inner,
            n_jobs=n_jobs_cv,
            est_params=M_best_params,
            method=self._predict_method["ml_M"],
        )

        W_inner = []
        for i, (train, test) in enumerate(smpls):
            M_iteration = M_hat["preds_inner"][i][train]
            M_iteration = np.clip(M_iteration, 1e-8, 1 - 1e-8)
            w = scipy.special.logit(M_iteration)
            W_inner.append(w)

        t_tune_res = _dml_tune(
            W_inner,
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

    @property
    def __smpls__inner(self):
        return self._smpls_inner[self._i_rep]

    def draw_sample_splitting(self):
        """
        Draw sample splitting for DoubleML models.

        The samples are drawn according to the attributes
        ``n_folds`` and ``n_rep``.

        Returns
        -------
        self : object
        """

        obj_dml_resampling = DoubleMLDoubleResampling(
            n_folds=self.n_folds,
            n_folds_inner=self.n_folds_inner,
            n_rep=self.n_rep,
            n_obs=self._dml_data.n_obs,
            stratify=self._strata,
        )
        self._smpls, self._smpls_inner = obj_dml_resampling.split_samples()

        return self

    def set_sample_splitting(self):
        raise NotImplementedError("set_sample_splitting is not implemented for DoubleMLLPLR.")

    def _compute_score(self, psi_elements, coef):
        if self.score == "nuisance_space":
            score_1 = psi_elements["y"] * np.exp(-coef * psi_elements["d"]) * psi_elements["d_tilde"]
            score = psi_elements["psi_hat"] * (score_1 - psi_elements["score_const"])
        elif self.score == "instrument":
            score = (psi_elements["y"] - scipy.special.expit(coef * psi_elements["d"] + psi_elements["r_hat"])) * psi_elements[
                "d_tilde"
            ]
        else:
            raise NotImplementedError

        return score

    def _compute_score_deriv(self, psi_elements, coef, inds=None):
        if self.score == "nuisance_space":
            deriv_1 = -psi_elements["y"] * np.exp(-coef * psi_elements["d"]) * psi_elements["d"]
            deriv = psi_elements["psi_hat"] * psi_elements["d_tilde"] * deriv_1
        elif self.score == "instrument":
            expit = scipy.special.expit(coef * psi_elements["d"] + psi_elements["r_hat"])
            deriv = -psi_elements["d"] * expit * (1 - expit) * psi_elements["d_tilde"]
        else:
            raise NotImplementedError

        return deriv
