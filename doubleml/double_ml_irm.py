import numpy as np
import pandas as pd
import warnings
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target

from .double_ml import DoubleML

from .double_ml_blp import DoubleMLBLP
from .double_ml_data import DoubleMLData
from .double_ml_score_mixins import LinearScoreMixin

from ._utils import _dml_cv_predict, _get_cond_smpls, _dml_tune, _check_finite_predictions, _check_is_propensity, \
    _trimm, _normalize_ipw, _check_score, _check_trimming


class DoubleMLIRM(LinearScoreMixin, DoubleML):
    """Double machine learning for interactive regression models

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    ml_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`g_0(D,X) = E[Y|X,D]`.
        For a binary outcome variable :math:`Y` (with values 0 and 1), a classifier implementing ``fit()`` and
        ``predict_proba()`` can also be specified. If :py:func:`sklearn.base.is_classifier` returns ``True``,
        ``predict_proba()`` is used otherwise ``predict()``.

    ml_m : classifier implementing ``fit()`` and ``predict_proba()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance function :math:`m_0(X) = E[D|X]`.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitons for the sample splitting.
        Default is ``1``.

    score : str or callable
        A str (``'ATE'`` or ``'ATTE'``) specifying the score function
        or a callable object / function with signature ``psi_a, psi_b = score(y, d, g_hat0, g_hat1, m_hat, smpls)``.
        Default is ``'ATE'``.

    dml_procedure : str
        A str (``'dml1'`` or ``'dml2'``) specifying the double machine learning algorithm.
        Default is ``'dml2'``.

    normalize_ipw : bool
        Indicates whether the inverse probability weights are normalized.
        Default is ``False``.

    trimming_rule : str
        A str (``'truncate'`` is the only choice) specifying the trimming approach.
        Default is ``'truncate'``.

    trimming_threshold : float
        The threshold used for trimming.
        Default is ``1e-2``.

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
    >>> from doubleml.datasets import make_irm_data
    >>> from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    >>> np.random.seed(3141)
    >>> ml_g = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
    >>> ml_m = RandomForestClassifier(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
    >>> data = make_irm_data(theta=0.5, n_obs=500, dim_x=20, return_type='DataFrame')
    >>> obj_dml_data = dml.DoubleMLData(data, 'y', 'd')
    >>> dml_irm_obj = dml.DoubleMLIRM(obj_dml_data, ml_g, ml_m)
    >>> dml_irm_obj.fit().summary
           coef   std err         t     P>|t|     2.5 %    97.5 %
    d  0.414073  0.238529  1.735941  0.082574 -0.053436  0.881581

    Notes
    -----
    **Interactive regression (IRM)** models take the form

    .. math::

        Y = g_0(D, X) + U, & &\\mathbb{E}(U | X, D) = 0,

        D = m_0(X) + V, & &\\mathbb{E}(V | X) = 0,

    where the treatment variable is binary, :math:`D \\in \\lbrace 0,1 \\rbrace`.
    We consider estimation of the average treatment effects when treatment effects are fully heterogeneous.
    Target parameters of interest in this model are the average treatment effect (ATE),

    .. math::

        \\theta_0 = \\mathbb{E}[g_0(1, X) - g_0(0,X)]

    and the average treatment effect of the treated (ATTE),

    .. math::

        \\theta_0 = \\mathbb{E}[g_0(1, X) - g_0(0,X) | D=1].
    """
    def __init__(self,
                 obj_dml_data,
                 ml_g,
                 ml_m,
                 n_folds=5,
                 n_rep=1,
                 score='ATE',
                 dml_procedure='dml2',
                 normalize_ipw=False,
                 trimming_rule='truncate',
                 trimming_threshold=1e-2,
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
        valid_scores = ['ATE', 'ATTE']
        _check_score(self.score, valid_scores, allow_callable=True)

        # set stratication for resampling
        self._strata = self._dml_data.d
        ml_g_is_classifier = self._check_learner(ml_g, 'ml_g', regressor=True, classifier=True)
        _ = self._check_learner(ml_m, 'ml_m', regressor=False, classifier=True)
        self._learner = {'ml_g': ml_g, 'ml_m': ml_m}
        self._normalize_ipw = normalize_ipw
        if ml_g_is_classifier:
            if obj_dml_data.binary_outcome:
                self._predict_method = {'ml_g': 'predict_proba', 'ml_m': 'predict_proba'}
            else:
                raise ValueError(f'The ml_g learner {str(ml_g)} was identified as classifier '
                                 'but the outcome variable is not binary with values 0 and 1.')
        else:
            self._predict_method = {'ml_g': 'predict', 'ml_m': 'predict_proba'}
        self._initialize_ml_nuisance_params()

        if not isinstance(self.normalize_ipw, bool):
            raise TypeError('Normalization indicator has to be boolean. ' +
                            f'Object of type {str(type(self.normalize_ipw))} passed.')
        self._trimming_rule = trimming_rule
        self._trimming_threshold = trimming_threshold
        _check_trimming(self._trimming_rule, self._trimming_threshold)

    @property
    def normalize_ipw(self):
        """
        Indicates whether the inverse probability weights are normalized.
        """
        return self._normalize_ipw

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
        valid_learner = ['ml_g0', 'ml_g1', 'ml_m']
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols}
                        for learner in valid_learner}

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError('The data must be of DoubleMLData type. '
                            f'{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed.')
        if obj_dml_data.z_cols is not None:
            raise ValueError('Incompatible data. ' +
                             ' and '.join(obj_dml_data.z_cols) +
                             ' have been set as instrumental variable(s). '
                             'To fit an interactive IV regression model use DoubleMLIIVM instead of DoubleMLIRM.')
        one_treat = (obj_dml_data.n_treat == 1)
        binary_treat = (type_of_target(obj_dml_data.d) == 'binary')
        zero_one_treat = np.all((np.power(obj_dml_data.d, 2) - obj_dml_data.d) == 0)
        if not (one_treat & binary_treat & zero_one_treat):
            raise ValueError('Incompatible data. '
                             'To fit an IRM model with DML '
                             'exactly one binary variable with values 0 and 1 '
                             'needs to be specified as treatment variable.')
        return

    def _nuisance_est(self, smpls, n_jobs_cv, return_models=False):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)
        # get train indices for d == 0 and d == 1
        smpls_d0, smpls_d1 = _get_cond_smpls(smpls, d)

        # nuisance g
        g_hat0 = _dml_cv_predict(self._learner['ml_g'], x, y, smpls=smpls_d0, n_jobs=n_jobs_cv,
                                 est_params=self._get_params('ml_g0'), method=self._predict_method['ml_g'],
                                 return_models=return_models)
        _check_finite_predictions(g_hat0['preds'], self._learner['ml_g'], 'ml_g', smpls)
        # adjust target values to consider only compatible subsamples
        g_hat0['targets'] = g_hat0['targets'].astype(float)
        g_hat0['targets'][d == 1] = np.nan

        if self._dml_data.binary_outcome:
            binary_preds = (type_of_target(g_hat0['preds']) == 'binary')
            zero_one_preds = np.all((np.power(g_hat0['preds'], 2) - g_hat0['preds']) == 0)
            if binary_preds & zero_one_preds:
                raise ValueError(f'For the binary outcome variable {self._dml_data.y_col}, '
                                 f'predictions obtained with the ml_g learner {str(self._learner["ml_g"])} are also '
                                 'observed to be binary with values 0 and 1. Make sure that for classifiers '
                                 'probabilities and not labels are predicted.')

        g_hat1 = {'preds': None, 'targets': None, 'models': None}
        if (self.score == 'ATE') | callable(self.score):
            g_hat1 = _dml_cv_predict(self._learner['ml_g'], x, y, smpls=smpls_d1, n_jobs=n_jobs_cv,
                                     est_params=self._get_params('ml_g1'), method=self._predict_method['ml_g'],
                                     return_models=return_models)
            _check_finite_predictions(g_hat1['preds'], self._learner['ml_g'], 'ml_g', smpls)
            # adjust target values to consider only compatible subsamples
            g_hat1['targets'] = g_hat1['targets'].astype(float)
            g_hat1['targets'][d == 0] = np.nan

            if self._dml_data.binary_outcome:
                binary_preds = (type_of_target(g_hat1['preds']) == 'binary')
                zero_one_preds = np.all((np.power(g_hat1['preds'], 2) - g_hat1['preds']) == 0)
                if binary_preds & zero_one_preds:
                    raise ValueError(f'For the binary outcome variable {self._dml_data.y_col}, '
                                     f'predictions obtained with the ml_g learner {str(self._learner["ml_g"])} are also '
                                     'observed to be binary with values 0 and 1. Make sure that for classifiers '
                                     'probabilities and not labels are predicted.')

        # nuisance m
        m_hat = _dml_cv_predict(self._learner['ml_m'], x, d, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_m'), method=self._predict_method['ml_m'],
                                return_models=return_models)
        _check_finite_predictions(m_hat['preds'], self._learner['ml_m'], 'ml_m', smpls)
        _check_is_propensity(m_hat['preds'], self._learner['ml_m'], 'ml_m', smpls, eps=1e-12)

        psi_a, psi_b = self._score_elements(y, d,
                                            g_hat0['preds'], g_hat1['preds'], m_hat['preds'],
                                            smpls)
        psi_elements = {'psi_a': psi_a,
                        'psi_b': psi_b}
        preds = {'predictions': {'ml_g0': g_hat0['preds'],
                                 'ml_g1': g_hat1['preds'],
                                 'ml_m': m_hat['preds']},
                 'targets': {'ml_g0': g_hat0['targets'],
                             'ml_g1': g_hat1['targets'],
                             'ml_m': m_hat['targets']},
                 'models': {'ml_g0': g_hat0['models'],
                            'ml_g1': g_hat1['models'],
                            'ml_m': m_hat['models']}
                 }

        return psi_elements, preds

    def _score_elements(self, y, d, g_hat0, g_hat1, m_hat, smpls):
        # fraction of treated for ATTE
        p_hat = None
        if self.score == 'ATTE':
            p_hat = np.full_like(d, np.nan, dtype='float64')
            for _, test_index in smpls:
                p_hat[test_index] = np.mean(d[test_index])

        m_hat = _trimm(m_hat, self.trimming_rule, self.trimming_threshold)

        if self.normalize_ipw:
            if self.dml_procedure == 'dml1':
                for _, test_index in smpls:
                    m_hat[test_index] = _normalize_ipw(m_hat[test_index], d[test_index])
            else:
                m_hat = _normalize_ipw(m_hat, d)

        # compute residuals
        u_hat0 = y - g_hat0
        u_hat1 = None
        if self.score == 'ATE':
            u_hat1 = y - g_hat1

        if isinstance(self.score, str):
            if self.score == 'ATE':
                psi_b = g_hat1 - g_hat0 \
                    + np.divide(np.multiply(d, u_hat1), m_hat) \
                    - np.divide(np.multiply(1.0-d, u_hat0), 1.0 - m_hat)
                psi_a = np.full_like(m_hat, -1.0)
            else:
                assert self.score == 'ATTE'
                psi_b = np.divide(np.multiply(d, u_hat0), p_hat) \
                    - np.divide(np.multiply(m_hat, np.multiply(1.0-d, u_hat0)),
                                np.multiply(p_hat, (1.0 - m_hat)))
                psi_a = - np.divide(d, p_hat)
        else:
            assert callable(self.score)
            psi_a, psi_b = self.score(y=y, d=d,
                                      g_hat0=g_hat0, g_hat1=g_hat1, m_hat=m_hat,
                                      smpls=smpls)

        return psi_a, psi_b

    def _nuisance_tuning(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                         search_mode, n_iter_randomized_search):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)
        # get train indices for d == 0 and d == 1
        smpls_d0, smpls_d1 = _get_cond_smpls(smpls, d)

        if scoring_methods is None:
            scoring_methods = {'ml_g': None,
                               'ml_m': None}

        train_inds = [train_index for (train_index, _) in smpls]
        train_inds_d0 = [train_index for (train_index, _) in smpls_d0]
        train_inds_d1 = [train_index for (train_index, _) in smpls_d1]
        g0_tune_res = _dml_tune(y, x, train_inds_d0,
                                self._learner['ml_g'], param_grids['ml_g'], scoring_methods['ml_g'],
                                n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)
        g1_tune_res = list()
        if self.score == 'ATE':
            g1_tune_res = _dml_tune(y, x, train_inds_d1,
                                    self._learner['ml_g'], param_grids['ml_g'], scoring_methods['ml_g'],
                                    n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        m_tune_res = _dml_tune(d, x, train_inds,
                               self._learner['ml_m'], param_grids['ml_m'], scoring_methods['ml_m'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        g0_best_params = [xx.best_params_ for xx in g0_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]
        if self.score == 'ATTE':
            params = {'ml_g0': g0_best_params,
                      'ml_m': m_best_params}
            tune_res = {'g0_tune': g0_tune_res,
                        'm_tune': m_tune_res}
        else:
            g1_best_params = [xx.best_params_ for xx in g1_tune_res]
            params = {'ml_g0': g0_best_params,
                      'ml_g1': g1_best_params,
                      'ml_m': m_best_params}
            tune_res = {'g0_tune': g0_tune_res,
                        'g1_tune': g1_tune_res,
                        'm_tune': m_tune_res}

        res = {'params': params,
               'tune_res': tune_res}

        return res

    def cate(self, basis):
        """
        Calculate conditional average treatment effects (CATE) for a given basis.

        Parameters
        ----------
        basis : :class:`pandas.DataFrame`
            The basis for estimating the best linear predictor. Has to have the shape ``(n_obs, d)``,
            where ``n_obs`` is the number of observations and ``d`` is the number of predictors.

        Returns
        -------
        model : :class:`doubleML.DoubleMLBLP`
            Best linear Predictor model.
        """
        valid_score = ['ATE']
        if self.score not in valid_score:
            raise ValueError('Invalid score ' + self.score + '. ' +
                             'Valid score ' + ' or '.join(valid_score) + '.')

        if self.n_rep != 1:
            raise NotImplementedError('Only implemented for one repetition. ' +
                                      f'Number of repetitions is {str(self.n_rep)}.')

        # define the orthogonal signal
        orth_signal = self.psi_elements['psi_b'].reshape(-1)
        # fit the best linear predictor
        model = DoubleMLBLP(orth_signal, basis=basis).fit()

        return model

    def gate(self, groups):
        """
        Calculate group average treatment effects (GATE) for mutually exclusive groups.

        Parameters
        ----------
        groups : :class:`pandas.DataFrame`
            The group indicator for estimating the best linear predictor.
            Has to be dummy coded with shape ``(n_obs, d)``, where ``n_obs`` is the number of observations
            and ``d`` is the number of groups or ``(n_obs, 1)`` and contain the corresponding groups (as str).

        Returns
        -------
        model : :class:`doubleML.DoubleMLBLPGATE`
            Best linear Predictor model for Group Effects.
        """
        valid_score = ['ATE']
        if self.score not in valid_score:
            raise ValueError('Invalid score ' + self.score + '. ' +
                             'Valid score ' + ' or '.join(valid_score) + '.')

        if self.n_rep != 1:
            raise NotImplementedError('Only implemented for one repetition. ' +
                                      f'Number of repetitions is {str(self.n_rep)}.')

        if not isinstance(groups, pd.DataFrame):
            raise TypeError('Groups must be of DataFrame type. '
                            f'Groups of type {str(type(groups))} was passed.')

        if not all(groups.dtypes == bool) or all(groups.dtypes == int):
            if groups.shape[1] == 1:
                groups = pd.get_dummies(groups, prefix='Group', prefix_sep='_')
            else:
                raise TypeError('Columns of groups must be of bool type or int type (dummy coded). '
                                'Alternatively, groups should only contain one column.')

        if any(groups.sum(0) <= 5):
            warnings.warn('At least one group effect is estimated with less than 6 observations.')

        # define the orthogonal signal
        orth_signal = self.psi_elements['psi_b'].reshape(-1)
        # fit the best linear predictor for GATE (different confint() method)
        model = DoubleMLBLP(orth_signal, basis=groups, is_gate=True).fit()

        return model
