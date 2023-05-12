import numpy as np
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target
from sklearn.base import clone

import warnings

from .double_ml import DoubleML
from .double_ml_data import DoubleMLData
from .double_ml_score_mixins import LinearScoreMixin
from ._utils import _dml_cv_predict, _dml_tune, _check_finite_predictions, _check_is_propensity, _check_score


class DoubleMLPLR(LinearScoreMixin, DoubleML):
    """Double machine learning for partially linear regression models

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

    dml_procedure : str
        A str (``'dml1'`` or ``'dml2'``) specifying the double machine learning algorithm.
        Default is ``'dml2'``.

    draw_sample_splitting : bool
        Indicates whether the sample splitting should be drawn during initialization of the object.
        Default is ``True``.

    apply_cross_fitting : bool
        Indicates whether cross-fitting should be applied.
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
                 ml_l,
                 ml_m,
                 ml_g=None,
                 n_folds=5,
                 n_rep=1,
                 score='partialling out',
                 dml_procedure='dml2',
                 draw_sample_splitting=True,
                 apply_cross_fitting=True):
        super().__init__(obj_dml_data,
                         n_folds,
                         n_rep,
                         score,
                         dml_procedure,
                         draw_sample_splitting,
                         apply_cross_fitting)

        self._check_data(self._dml_data)
        valid_scores = ['IV-type', 'partialling out']
        _check_score(self.score, valid_scores, allow_callable=True)

        _ = self._check_learner(ml_l, 'ml_l', regressor=True, classifier=False)
        ml_m_is_classifier = self._check_learner(ml_m, 'ml_m', regressor=True, classifier=True)
        self._learner = {'ml_l': ml_l, 'ml_m': ml_m}

        if ml_g is not None:
            if (isinstance(self.score, str) & (self.score == 'IV-type')) | callable(self.score):
                _ = self._check_learner(ml_g, 'ml_g', regressor=True, classifier=False)
                self._learner['ml_g'] = ml_g
            else:
                assert (isinstance(self.score, str) & (self.score == 'partialling out'))
                warnings.warn(('A learner ml_g has been provided for score = "partialling out" but will be ignored. "'
                               'A learner ml_g is not required for estimation.'))
        elif isinstance(self.score, str) & (self.score == 'IV-type'):
            warnings.warn(("For score = 'IV-type', learners ml_l and ml_g should be specified. "
                           "Set ml_g = clone(ml_l)."))
            self._learner['ml_g'] = clone(ml_l)

        self._predict_method = {'ml_l': 'predict'}
        if 'ml_g' in self._learner:
            self._predict_method['ml_g'] = 'predict'
        if ml_m_is_classifier:
            if self._dml_data.binary_treats.all():
                self._predict_method['ml_m'] = 'predict_proba'
            else:
                raise ValueError(f'The ml_m learner {str(ml_m)} was identified as classifier '
                                 'but at least one treatment variable is not binary with values 0 and 1.')
        else:
            self._predict_method['ml_m'] = 'predict'

        self._initialize_ml_nuisance_params()

    def _initialize_ml_nuisance_params(self):
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols}
                        for learner in self._learner}

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError('The data must be of DoubleMLData type. '
                            f'{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed.')
        if obj_dml_data.z_cols is not None:
            raise ValueError('Incompatible data. ' +
                             ' and '.join(obj_dml_data.z_cols) +
                             ' have been set as instrumental variable(s). '
                             'To fit a partially linear IV regression model use DoubleMLPLIV instead of DoubleMLPLR.')
        return

    def _nuisance_est(self, smpls, n_jobs_cv, return_models=False):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)

        # nuisance l
        l_hat = _dml_cv_predict(self._learner['ml_l'], x, y, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_l'), method=self._predict_method['ml_l'],
                                return_models=return_models)
        _check_finite_predictions(l_hat['preds'], self._learner['ml_l'], 'ml_l', smpls)

        # nuisance m
        m_hat = _dml_cv_predict(self._learner['ml_m'], x, d, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_m'), method=self._predict_method['ml_m'],
                                return_models=return_models)
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

        # an estimate of g is obtained for the IV-type score and callable scores
        g_hat = {'preds': None, 'targets': None, 'models': None}
        if 'ml_g' in self._learner:
            # get an initial estimate for theta using the partialling out score
            psi_a = -np.multiply(d - m_hat['preds'], d - m_hat['preds'])
            psi_b = np.multiply(d - m_hat['preds'], y - l_hat['preds'])
            theta_initial = -np.nanmean(psi_b) / np.nanmean(psi_a)
            # nuisance g
            g_hat = _dml_cv_predict(self._learner['ml_g'], x, y - theta_initial*d, smpls=smpls, n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_g'), method=self._predict_method['ml_g'],
                                    return_models=return_models)
            _check_finite_predictions(g_hat['preds'], self._learner['ml_g'], 'ml_g', smpls)

        psi_a, psi_b = self._score_elements(y, d, l_hat['preds'], m_hat['preds'], g_hat['preds'], smpls)
        psi_elements = {'psi_a': psi_a,
                        'psi_b': psi_b}
        preds = {'predictions': {'ml_l': l_hat['preds'],
                                 'ml_m': m_hat['preds'],
                                 'ml_g': g_hat['preds']},
                 'targets': {'ml_l': l_hat['targets'],
                             'ml_m': m_hat['targets'],
                             'ml_g': g_hat['targets']},
                 'models': {'ml_l': l_hat['models'],
                            'ml_m': m_hat['models'],
                            'ml_g': g_hat['models']}}

        return psi_elements, preds

    def _score_elements(self, y, d, l_hat, m_hat, g_hat, smpls):
        # compute residuals
        u_hat = y - l_hat
        v_hat = d - m_hat

        if isinstance(self.score, str):
            if self.score == 'IV-type':
                psi_a = - np.multiply(v_hat, d)
                psi_b = np.multiply(v_hat, y - g_hat)
            else:
                assert self.score == 'partialling out'
                psi_a = -np.multiply(v_hat, v_hat)
                psi_b = np.multiply(v_hat, u_hat)
        else:
            assert callable(self.score)
            psi_a, psi_b = self.score(y=y, d=d,
                                      l_hat=l_hat, m_hat=m_hat, g_hat=g_hat,
                                      smpls=smpls)

        return psi_a, psi_b

    def _nuisance_tuning(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                         search_mode, n_iter_randomized_search):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)

        if scoring_methods is None:
            scoring_methods = {'ml_l': None,
                               'ml_m': None,
                               'ml_g': None}

        train_inds = [train_index for (train_index, _) in smpls]
        l_tune_res = _dml_tune(y, x, train_inds,
                               self._learner['ml_l'], param_grids['ml_l'], scoring_methods['ml_l'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)
        m_tune_res = _dml_tune(d, x, train_inds,
                               self._learner['ml_m'], param_grids['ml_m'], scoring_methods['ml_m'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        l_best_params = [xx.best_params_ for xx in l_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]

        # an ML model for g is obtained for the IV-type score and callable scores
        if 'ml_g' in self._learner:
            # construct an initial theta estimate from the tuned models using the partialling out score
            l_hat = np.full_like(y, np.nan)
            m_hat = np.full_like(d, np.nan)
            for idx, (train_index, _) in enumerate(smpls):
                l_hat[train_index] = l_tune_res[idx].predict(x[train_index, :])
                m_hat[train_index] = m_tune_res[idx].predict(x[train_index, :])
            psi_a = -np.multiply(d - m_hat, d - m_hat)
            psi_b = np.multiply(d - m_hat, y - l_hat)
            theta_initial = -np.nanmean(psi_b) / np.nanmean(psi_a)
            g_tune_res = _dml_tune(y - theta_initial*d, x, train_inds,
                                   self._learner['ml_g'], param_grids['ml_g'], scoring_methods['ml_g'],
                                   n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

            g_best_params = [xx.best_params_ for xx in g_tune_res]
            params = {'ml_l': l_best_params,
                      'ml_m': m_best_params,
                      'ml_g': g_best_params}
            tune_res = {'l_tune': l_tune_res,
                        'm_tune': m_tune_res,
                        'g_tune': g_tune_res}
        else:
            params = {'ml_l': l_best_params,
                      'ml_m': m_best_params}
            tune_res = {'l_tune': l_tune_res,
                        'm_tune': m_tune_res}

        res = {'params': params,
               'tune_res': tune_res}

        return res
