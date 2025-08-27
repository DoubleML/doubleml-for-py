import inspect

import numpy as np
from torch.sparse import sampled_addmm

from doubleml.utils._estimation import (
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
from sklearn.base import clone
from sklearn.utils import check_X_y
import scipy
from sklearn.utils.multiclass import type_of_target

from doubleml import DoubleMLData
from doubleml.double_ml import DoubleML
from doubleml.double_ml_score_mixins import NonLinearScoreMixin
from doubleml.utils import DoubleMLClusterResampling
from doubleml.utils._checks import _check_score, _check_finite_predictions, _check_is_propensity
from doubleml.utils.resampling import DoubleMLDoubleResampling





class DoubleMLLogit(NonLinearScoreMixin, DoubleML):
    """Double machine learning for partially linear regression models

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    ml_r : estimator implementing ``fit()`` and ``predict()``
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
        A str (``'nuisance_space'`` or ``'instrument'``) specifying the score function
        or a callable object / function with signature ``psi_a, psi_b = score(y, d, l_hat, m_hat, g_hat, smpls)``.
        Default is ``'partialling out'``.

    draw_sample_splitting : bool
        Indicates whether the sample splitting should be drawn during initialization of the object.
        Default is ``True``.

    Examples
    --------
    >>> import numpy as np
    >>> import doubleml as dml
    >>> from doubleml.datasets import make_plr_CCDDHNR2018
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.base import clone
    >>> np.random.seed(3141)
    >>> learner = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
    >>> ml_g = learner
    >>> ml_m = learner
    >>> obj_dml_data = make_plr_CCDDHNR2018(alpha=0.5, n_obs=500, dim_x=20)
    >>> dml_plr_obj = dml.DoubleMLPLR(obj_dml_data, ml_g, ml_m)
    >>> dml_plr_obj.fit().summary
           coef  std err          t         P>|t|     2.5 %    97.5 %
    d  0.462321  0.04107  11.256983  2.139582e-29  0.381826  0.542816

    Notes
    -----
    **Partially linear regression (PLR)** models take the form

    .. math::

        Y = D \\theta_0 + g_0(X) + \\zeta, & &\\mathbb{E}(\\zeta | D,X) = 0,

        D = m_0(X) + V, & &\\mathbb{E}(V | X) = 0,

    where :math:`Y` is the outcome variable and :math:`D` is the policy variable of interest.
    The high-dimensional vector :math:`X = (X_1, \\ldots, X_p)` consists of other confounding covariates,
    and :math:`\\zeta` and :math:`V` are stochastic errors.
    """

    def __init__(self,
                 obj_dml_data,
                 ml_M,
                 ml_t,
                 ml_m,
                 ml_a=None,
                 n_folds=5,
                 n_folds_inner=5,
                 n_rep=1,
                 score='nuisance_space',
                 draw_sample_splitting=True):
        self.n_folds_inner = n_folds_inner
        super().__init__(obj_dml_data,
                         n_folds,
                         n_rep,
                         score,
                         draw_sample_splitting)
        self._coef_bounds = (-1e-2, 1e2)
        self._coef_start_val = 1.0

        self._check_data(self._dml_data)
        valid_scores = ['nuisance_space', 'instrument']
        _check_score(self.score, valid_scores, allow_callable=True)

        _ = self._check_learner(ml_t, 'ml_t', regressor=True, classifier=False)
        _ = self._check_learner(ml_M, 'ml_M', regressor=False, classifier=True)

        if not np.array_equal(np.unique(obj_dml_data.y), [0, 1]):
            ml_m_is_classifier = self._check_learner(ml_m, 'ml_m', regressor=False, classifier=True)
        else:
            ml_m_is_classifier = self._check_learner(ml_m, 'ml_m', regressor=True, classifier=False)
        self._learner = {'ml_m': ml_m, 'ml_t': ml_t, 'ml_M': ml_M}

        if ml_a is not None:
            ml_a_is_classifier = self._check_learner(ml_a, 'ml_a', regressor=True, classifier=True)
            self._learner['ml_a'] = ml_a
        else:
            self._learner['ml_a'] = clone(ml_m)
            ml_a_is_classifier = ml_m_is_classifier

        self._predict_method = {'ml_t': 'predict', 'ml_M': 'predict_proba'}

        if ml_m_is_classifier:
            if self._dml_data.binary_treats.all():
                self._predict_method['ml_m'] = 'predict_proba'
            else:
                raise ValueError(f'The ml_m learner {str(ml_m)} was identified as classifier '
                                 'but at least one treatment variable is not binary with values 0 and 1.')
        else:
            self._predict_method['ml_m'] = 'predict'

        if ml_a_is_classifier:
            if self._dml_data.binary_treats.all():
                self._predict_method['ml_a'] = 'predict_proba'
            else:
                raise ValueError(f'The ml_a learner {str(ml_a)} was identified as classifier '
                                 'but at least one treatment variable is not binary with values 0 and 1.')
        else:
            self._predict_method['ml_a'] = 'predict'

        if score == 'instrument':
            sig = inspect.signature(self.learner['ml_a'].fit)
            if not 'sample_weight' in sig.parameters:
                raise ValueError('Learner \"ml_a\" who supports sample_weight is required for score type \"instrument\"')

        self._initialize_ml_nuisance_params()
        self._external_predictions_implemented = True

    def _initialize_ml_nuisance_params(self):
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols}
                        for learner in self._learner}

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError('The data must be of DoubleMLData type. '
                            f'{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed.')
        if not np.array_equal(np.unique(obj_dml_data.y), [0, 1]):
            raise TypeError('The outcome variable y must be binary with values 0 and 1.')
        return


    def _double_dml_cv_predict(self, estimator, estimator_name,  x, y, smpls=None, smpls_inner=None,
                    n_jobs=None, est_params=None, method='predict', sample_weights=None):
        res = {}
        res['preds'] = np.zeros(y.shape, dtype=float)
        res['preds_inner'] = []
        res['models'] = []
        for smpls_single_split, smpls_double_split in zip(smpls, smpls_inner):
            res_inner = _dml_cv_predict(estimator, x, y, smpls=smpls_double_split, n_jobs=n_jobs,
                                    est_params=est_params, method=method,
                                    return_models=True, smpls_is_partition=True, sample_weights=sample_weights)
            _check_finite_predictions(res_inner['preds'], estimator, estimator_name, smpls_double_split)

            res['preds_inner'].append(res_inner['preds'])
            for model in res_inner['models']:
                res['models'].append(model)
                if method == 'predict_proba':
                    res['preds'][smpls_single_split[1]] += model.predict_proba(x[smpls_single_split[1]])[:, 1]
                else:
                    res['preds'][smpls_single_split[1]] += model.predict(x[smpls_single_split[1]])
        res["preds_inner"]
        res["preds"] /= len(smpls)
        res['targets'] = np.copy(y)
        return res



    def _nuisance_est(self, smpls, n_jobs_cv, external_predictions, return_models=False):
        # TODO: How to deal with smpls_inner?
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         ensure_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         ensure_all_finite=False)
        x_d_concat = np.hstack((d.reshape(-1,1), x))
        m_external = external_predictions['ml_m'] is not None
        M_external = external_predictions['ml_M'] is not None
        t_external = external_predictions['ml_t'] is not None
        if 'ml_a' in self._learner:
            a_external = external_predictions['ml_a'] is not None
        else:
            a_external = False

        if M_external:
            M_hat = {'preds': external_predictions['ml_M'],
                     'targets': None,
                     'models': None}
        else:
            M_hat = (self._double_dml_cv_predict(self._learner['ml_M'], 'ml_M', x_d_concat, y, smpls=smpls, smpls_inner=self.__smpls__inner,
                                                n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_M'), method=self._predict_method['ml_M']))


        # nuisance m
        if m_external:
            m_hat = {'preds': external_predictions['ml_m'],
                     'targets': None,
                     'models': None}
        else:
            if self.score == 'instrument':
                weights = []
                for i, (train, test) in enumerate(smpls):
                    weights.append( M_hat['preds_inner'][i][train] * (1-M_hat['preds_inner'][i][train]))
                m_hat = _dml_cv_predict(self._learner['ml_m'], x, d, smpls=smpls, n_jobs=n_jobs_cv,
                                        est_params=self._get_params('ml_m'), method=self._predict_method['ml_m'],
                                        return_models=return_models, weights=weights)

            elif self.score == 'nuisance_space':
                filtered_smpls = []
                for train, test in smpls:
                    train_filtered = train[y[train] == 0]
                    filtered_smpls.append((train_filtered, test))
                m_hat = _dml_cv_predict(self._learner['ml_m'], x, d, smpls=filtered_smpls, n_jobs=n_jobs_cv,
                                        est_params=self._get_params('ml_m'), method=self._predict_method['ml_m'],
                                        return_models=return_models)
            else:
                raise NotImplementedError
            _check_finite_predictions(m_hat['preds'], self._learner['ml_m'], 'ml_m', smpls)

        if self._check_learner(self._learner['ml_m'], 'ml_m', regressor=True, classifier=True):
            _check_is_propensity(m_hat['preds'], self._learner['ml_m'], 'ml_m', smpls, eps=1e-12)

        if self._dml_data.binary_treats[self._dml_data.d_cols[self._i_treat]]:
            binary_preds = (type_of_target(m_hat['preds']) == 'binary')
            zero_one_preds = np.all((np.power(m_hat['preds'], 2) - m_hat['preds']) == 0)
            if binary_preds & zero_one_preds:
                raise ValueError(f'For the binary treatment variable {self._dml_data.d_cols[self._i_treat]}, '
                                 f'predictions obtained with the ml_m learner {str(self._learner["ml_m"])} are also '
                                 'observed to be binary with values 0 and 1. Make sure that for classifiers '
                                 'probabilities and not labels are predicted.')




        if a_external:
            a_hat = {'preds': external_predictions['ml_a'],
                     'targets': None,
                     'models': None}
        else:
            a_hat = (self._double_dml_cv_predict(self._learner['ml_a'], 'ml_a', x, d, smpls=smpls, smpls_inner=self.__smpls__inner,
                                                n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_a'), method=self._predict_method['ml_a']))


        r_legacy = np.zeros_like(y)
        smpls_inner = self.__smpls__inner
        M_hat_l = {}
        a_hat_l = {}
        M_hat_l['preds_inner'] = []
        M_hat_l['preds'] = np.full_like(y, np.nan)
        a_hat_l['preds_inner'] = []
        a_hat_l['preds'] = np.full_like(y, np.nan)
        for smpls_single_split, smpls_double_split in zip(smpls, smpls_inner):
            test = smpls_single_split[1]
            train = smpls_single_split[0]
            # r_legacy[test] =
            Mleg, aleg, a_nf_leg = self.legacy_implementation(y[train], x[train], d[train], x[test], d[test],
                                                              self._learner['ml_m'], self._learner['ml_M'],
                                                              smpls_single_split, smpls_double_split, y, x, d,
                                                              x_d_concat, n_jobs_cv)
            Mtemp = np.full_like(y, np.nan)
            Mtemp[train] = Mleg
            Atemp = np.full_like(y, np.nan)
            Atemp[train] = aleg
            M_hat_l['preds_inner'].append(Mtemp)
            a_hat_l['preds_inner'].append(Atemp)
            a_hat_l['preds'][test] = a_nf_leg

        #r_hat['preds'] = r_legacy



        W_inner = []
        beta = np.zeros(d.shape, dtype=float)

        for i, (train, test) in enumerate(smpls):
            M_iteration = M_hat['preds_inner'][i][train]
            M_iteration = np.clip(M_iteration, 1e-8, 1 - 1e-8)
            w = scipy.special.logit(M_iteration)
            W_inner.append(w)
            d_tilde = (d - a_hat['preds_inner'][i])[train]
            beta[test] = np.sum(d_tilde * w) / np.sum(d_tilde ** 2)


        # nuisance t
        if t_external:
            t_hat = {'preds': external_predictions['ml_t'],
                     'targets': None,
                     'models': None}
        else:
            t_hat = _dml_cv_predict(self._learner['ml_t'], x, W_inner, smpls=smpls, n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_t'), method=self._predict_method['ml_t'],
                                    return_models=return_models)
            _check_finite_predictions(t_hat['preds'], self._learner['ml_t'], 'ml_t', smpls)


        r_hat = {}
        r_hat['preds'] = t_hat['preds'] - beta * a_hat['preds']

        psi_elements = self._score_elements(y, d, r_hat['preds'], m_hat['preds'])

        preds = {'predictions': {'ml_r': r_hat['preds'],
                                 'ml_m': m_hat['preds'],
                                 'ml_a': a_hat['preds'],
                                 'ml_t': t_hat['preds'],
                                 'ml_M': M_hat['preds']},
                 'targets': {'ml_r': None,
                             'ml_m': m_hat['targets'],
                             'ml_a': a_hat['targets'],
                             'ml_t': t_hat['targets'],
                             'ml_M': M_hat['targets']},
                 'models': {'ml_r': None,
                            'ml_m': m_hat['models'],
                            'ml_a': a_hat['models'],
                            'ml_t': t_hat['models'],
                            'ml_M': M_hat['models']}}

        return psi_elements, preds


    def legacy_implementation(self, Yfold: np.ndarray, Xfold: np.ndarray, Afold: np.ndarray, XnotFold: np.ndarray, AnotFold: np.ndarray,
                    learner, learnerClassifier, smpls_single_split, smpls_double_split, yfull, xfull, afull, x_d_concat, n_jobs_cv, noFolds: int = 5, seed=None, )-> (np.ndarray, np.ndarray, np.ndarray):

        def learn_predict(X, Y, Xpredict, learner, learnerClassifier, fit_args={}):
            results = []
            if len(np.unique(Y)) == 2:
                learnerClassifier.fit(X, Y, **fit_args)
                for x in Xpredict:
                    results.append(learnerClassifier.predict_proba(x)[:, 1])
            else:
                learner.fit(X, Y, **fit_args)
                for x in Xpredict:
                    results.append(learner.predict(x))
            return (*results,)

        nFold = len(Yfold)
        i = np.remainder(np.arange(nFold), noFolds)
        np.random.default_rng(seed).shuffle(i)

        M = np.zeros((nFold))
        a_hat = np.zeros((nFold))
        a_hat_notFold = np.zeros((len(XnotFold)))
        M_notFold = np.zeros((len(XnotFold)))
        loss = {}

        a_hat_inner = _dml_cv_predict(self._learner['ml_a'], xfull, afull, smpls=smpls_double_split, n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_a'), method=self._predict_method['ml_a'],
                                    return_models=True, smpls_is_partition=True)
        _check_finite_predictions(a_hat_inner['preds'], self._learner['ml_a'], 'ml_a', smpls_double_split)
        a_hat_notFold = np.full_like(yfull, 0.)
        for model in a_hat_inner['models']:
            if self._predict_method['ml_a'] == 'predict_proba':
                a_hat_notFold[smpls_single_split[1]] += model.predict_proba(xfull[smpls_single_split[1]])[:, 1]
            else:
                a_hat_notFold[smpls_single_split[1]] += model.predict(xfull[smpls_single_split[1]])

        M_hat = _dml_cv_predict(self._learner['ml_M'], x_d_concat, yfull, smpls=smpls_double_split, n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_M'), method=self._predict_method['ml_M'],
                                    return_models=True, smpls_is_partition=True)
        _check_finite_predictions(M_hat['preds'], self._learner['ml_M'], 'ml_M', smpls_double_split)

        M = M_hat['preds'][~np.isnan(M_hat['preds'])]
        a_hat = a_hat_inner['preds'][~np.isnan(a_hat_inner['preds'])]
        a_hat_notFold = a_hat_notFold[smpls_single_split[1]]

        np.clip(M, 1e-8, 1 - 1e-8, out=M)
#        loss["M"] = compute_loss(Yfold, M)
#        loss["a_hat"] = compute_loss(Afold, a_hat)
        a_hat_notFold /= noFolds
      #  M_notFold /= noFolds
        np.clip(M_notFold, 1e-8, 1 - 1e-8, out=M_notFold)

        # Obtain preliminary estimate of beta based on M and residual of a
        W = scipy.special.logit(M)
        A_resid = Afold - a_hat
        beta_notFold = sum(A_resid * W) / sum(A_resid ** 2)
    #    print(beta_notFold)
        t_notFold, = learn_predict(Xfold, W, [XnotFold], learner, learnerClassifier)
        W_notFold = scipy.special.expit(M_notFold)
#        loss["t"] = compute_loss(W_notFold, t_notFold)


        # Compute r based on estimates for W=logit(M), beta and residual of A
        r_notFold = t_notFold - beta_notFold * a_hat_notFold

        return M, a_hat, a_hat_notFold #r_notFold #, a_hat_notFold, M_notFold, t_notFold

    def _score_elements(self, y, d, r_hat, m_hat):
        # compute residual
        d_tilde = d - m_hat
        psi_hat = scipy.special.expit(-r_hat)
        score_const = d_tilde * (1 - y) * np.exp(r_hat)
        psi_elements = {"y": y, "d": d, "d_tilde": d_tilde, "r_hat": r_hat, "m_hat": m_hat, "psi_hat": psi_hat, "score_const": score_const}

        return psi_elements

    @property
    def _score_element_names(self):
        return ['y', 'd', 'd_tilde', 'r_hat', 'm_hat', 'psi_hat', 'score_const']

    def _sensitivity_element_est(self, preds):
       pass

    def _nuisance_tuning(self):
        pass

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

        obj_dml_resampling = DoubleMLDoubleResampling(n_folds=self.n_folds,
                                                      n_folds_inner=self.n_folds_inner,
                                                      n_rep=self.n_rep,
                                                      n_obs=self._dml_data.n_obs,
                                                      stratify=self._strata)
        self._smpls, self._smpls_inner = obj_dml_resampling.split_samples()

        return self

    def set_sample_splitting(self):
        raise NotImplementedError('set_sample_splitting is not implemented for DoubleMLLogit.')

    def _compute_score(self, psi_elements, coef):

        if self.score == 'nuisance_space':
            score_1 = psi_elements["y"] * np.exp(-coef * psi_elements["d"]) * psi_elements["d_tilde"]
            score = psi_elements["psi_hat"] * (score_1 - psi_elements["score_const"])
        elif self.score == 'instrument':
            score = (psi_elements["y"] - np.exp(coef * psi_elements["d"]+ psi_elements["r_hat"])) * psi_elements["d_tilde"]
        else:
            raise NotImplementedError

        return score

    def _compute_score_deriv(self, psi_elements, coef, inds=None):
        if self.score == 'nuisance_space':
            deriv_1 = - psi_elements["y"] * np.exp(-coef * psi_elements["d"]) * psi_elements["d"]
            deriv = psi_elements["psi_hat"] * psi_elements["d_tilde"] *  deriv_1
        elif self.score == 'instrument':
            deriv = - psi_elements["d"] * np.exp(coef * psi_elements["d"]+ psi_elements["r_hat"]) * psi_elements["d_tilde"]
        else:
            raise NotImplementedError

        return deriv