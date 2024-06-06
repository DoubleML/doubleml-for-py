import numpy as np
import pandas as pd
import warnings
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target

from ..double_ml import DoubleML

from ..utils.blp import DoubleMLBLP
from ..utils.policytree import DoubleMLPolicyTree
from ..double_ml_data import DoubleMLData
from ..double_ml_score_mixins import LinearScoreMixin

from ..utils._estimation import _dml_cv_predict, _get_cond_smpls, _dml_tune, _trimm, _normalize_ipw, _cond_targets
from ..utils._checks import _check_score, _check_trimming, _check_finite_predictions, _check_is_propensity, _check_integer, \
    _check_weights


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

    weights : array, dict or None
        An numpy array of weights for each individual observation. If None, then the ``'ATE'`` score
        is applied (corresponds to weights equal to 1). Can only be used with ``score = 'ATE'``.
        An array has to be of shape ``(n,)``, where ``n`` is the number of observations.
        A dictionary can be used to specify weights which depend on the treatment variable.
        In this case, the dictionary has to contain two keys ``weights`` and ``weights_bar``, where the values
        have to be arrays of shape ``(n,)`` and ``(n, n_rep)``.
        Default is ``None``.

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
                 weights=None,
                 normalize_ipw=False,
                 trimming_rule='truncate',
                 trimming_threshold=1e-2,
                 draw_sample_splitting=True):
        super().__init__(obj_dml_data,
                         n_folds,
                         n_rep,
                         score,
                         draw_sample_splitting)

        self._check_data(self._dml_data)
        valid_scores = ['ATE', 'ATTE']
        _check_score(self.score, valid_scores, allow_callable=True)

        # set stratication for resampling
        self._strata = self._dml_data.d
        if draw_sample_splitting:
            self.draw_sample_splitting()

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

        self._sensitivity_implemented = True
        self._external_predictions_implemented = True

        _check_weights(weights, score, obj_dml_data.n_obs, self.n_rep)
        self._initialize_weights(weights)

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

    @property
    def weights(self):
        """
        Specifies the weights for a weighted ATE.
        """
        return self._weights

    def _initialize_ml_nuisance_params(self):
        valid_learner = ['ml_g0', 'ml_g1', 'ml_m']
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols}
                        for learner in valid_learner}

    def _initialize_weights(self, weights):
        if weights is None:
            weights = np.ones(self._dml_data.n_obs)
        if isinstance(weights, np.ndarray):
            self._weights = {'weights': weights}
        else:
            assert isinstance(weights, dict)
            self._weights = weights

    def _get_weights(self, m_hat=None):
        # standard case for ATE
        if self.score == 'ATE':
            weights = self._weights['weights']
            if 'weights_bar' not in self._weights.keys():
                weights_bar = self._weights['weights']
            else:
                weights_bar = self._weights['weights_bar'][:, self._i_rep]
        else:
            # special case for ATTE
            assert self.score == 'ATTE'
            assert m_hat is not None
            subgroup = self._weights['weights'] * self._dml_data.d
            subgroup_probability = np.mean(subgroup)
            weights = np.divide(subgroup, subgroup_probability)

            weights_bar = np.divide(
                np.multiply(m_hat, self._weights['weights']),
                subgroup_probability)

        return weights, weights_bar

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

    def _nuisance_est(self, smpls, n_jobs_cv, external_predictions, return_models=False):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)
        # get train indices for d == 0 and d == 1
        smpls_d0, smpls_d1 = _get_cond_smpls(smpls, d)
        g0_external = external_predictions['ml_g0'] is not None
        g1_external = external_predictions['ml_g1'] is not None
        m_external = external_predictions['ml_m'] is not None

        # nuisance g
        if g0_external:
            # use external predictions
            g_hat0 = {'preds': external_predictions['ml_g0'],
                      'targets': None,
                      'models': None}
        else:
            g_hat0 = _dml_cv_predict(self._learner['ml_g'], x, y, smpls=smpls_d0, n_jobs=n_jobs_cv,
                                     est_params=self._get_params('ml_g0'), method=self._predict_method['ml_g'],
                                     return_models=return_models)
            _check_finite_predictions(g_hat0['preds'], self._learner['ml_g'], 'ml_g', smpls)
            g_hat0['targets'] = _cond_targets(g_hat0['targets'], cond_sample=(d == 0))

            if self._dml_data.binary_outcome:
                binary_preds = (type_of_target(g_hat0['preds']) == 'binary')
                zero_one_preds = np.all((np.power(g_hat0['preds'], 2) - g_hat0['preds']) == 0)
                if binary_preds & zero_one_preds:
                    raise ValueError(f'For the binary outcome variable {self._dml_data.y_col}, '
                                     f'predictions obtained with the ml_g learner {str(self._learner["ml_g"])} are also '
                                     'observed to be binary with values 0 and 1. Make sure that for classifiers '
                                     'probabilities and not labels are predicted.')

        if g1_external:
            # use external predictions
            g_hat1 = {'preds': external_predictions['ml_g1'],
                      'targets': None,
                      'models': None}
        else:
            g_hat1 = _dml_cv_predict(self._learner['ml_g'], x, y, smpls=smpls_d1, n_jobs=n_jobs_cv,
                                     est_params=self._get_params('ml_g1'), method=self._predict_method['ml_g'],
                                     return_models=return_models)
            _check_finite_predictions(g_hat1['preds'], self._learner['ml_g'], 'ml_g', smpls)
            # adjust target values to consider only compatible subsamples
            g_hat1['targets'] = _cond_targets(g_hat1['targets'], cond_sample=(d == 1))

        if self._dml_data.binary_outcome & (self.score != 'ATTE'):
            binary_preds = (type_of_target(g_hat1['preds']) == 'binary')
            zero_one_preds = np.all((np.power(g_hat1['preds'], 2) - g_hat1['preds']) == 0)
            if binary_preds & zero_one_preds:
                raise ValueError(f'For the binary outcome variable {self._dml_data.y_col}, '
                                 f'predictions obtained with the ml_g learner {str(self._learner["ml_g"])} are also '
                                 'observed to be binary with values 0 and 1. Make sure that for classifiers '
                                 'probabilities and not labels are predicted.')

        # nuisance m
        if m_external:
            # use external predictions
            m_hat = {'preds': external_predictions['ml_m'],
                     'targets': None,
                     'models': None}
        else:
            m_hat = _dml_cv_predict(self._learner['ml_m'], x, d, smpls=smpls, n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_m'), method=self._predict_method['ml_m'],
                                    return_models=return_models)
            _check_finite_predictions(m_hat['preds'], self._learner['ml_m'], 'ml_m', smpls)
            _check_is_propensity(m_hat['preds'], self._learner['ml_m'], 'ml_m', smpls, eps=1e-12)
        # also trimm external predictions
        m_hat['preds'] = _trimm(m_hat['preds'], self.trimming_rule, self.trimming_threshold)

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

        m_hat_adj = np.full_like(m_hat, np.nan, dtype='float64')
        if self.normalize_ipw:
            m_hat_adj = _normalize_ipw(m_hat, d)
        else:
            m_hat_adj = m_hat

        # compute residuals
        u_hat0 = y - g_hat0
        u_hat1 = y - g_hat1

        if (self.score == 'ATE') or (self.score == 'ATTE'):
            weights, weights_bar = self._get_weights(m_hat=m_hat_adj)
            psi_b = weights * (g_hat1 - g_hat0) \
                + weights_bar * (
                    np.divide(np.multiply(d, u_hat1), m_hat_adj)
                    - np.divide(np.multiply(1.0-d, u_hat0), 1.0 - m_hat_adj))
            if self.score == 'ATE':
                psi_a = np.full_like(m_hat_adj, -1.0)
            else:
                assert self.score == 'ATTE'
                psi_a = -1.0 * weights
        else:
            assert callable(self.score)
            psi_a, psi_b = self.score(y=y, d=d,
                                      g_hat0=g_hat0, g_hat1=g_hat1, m_hat=m_hat_adj,
                                      smpls=smpls)

        return psi_a, psi_b

    def _sensitivity_element_est(self, preds):
        # set elments for readability
        y = self._dml_data.y
        d = self._dml_data.d

        m_hat = preds['predictions']['ml_m']
        g_hat0 = preds['predictions']['ml_g0']
        if self.score == 'ATE':
            g_hat1 = preds['predictions']['ml_g1']
        else:
            assert self.score == 'ATTE'
            g_hat1 = y

        # use weights make this extendable
        weights, weights_bar = self._get_weights(m_hat=m_hat)

        sigma2_score_element = np.square(y - np.multiply(d, g_hat1) - np.multiply(1.0-d, g_hat0))
        sigma2 = np.mean(sigma2_score_element)
        psi_sigma2 = sigma2_score_element - sigma2

        # calc m(W,alpha) and Riesz representer
        m_alpha = np.multiply(weights, np.multiply(weights_bar, (np.divide(1.0, m_hat) + np.divide(1.0, 1.0-m_hat))))
        rr = np.multiply(weights_bar, (np.divide(d, m_hat) - np.divide(1.0-d, 1.0-m_hat)))

        nu2_score_element = np.multiply(2.0, m_alpha) - np.square(rr)
        nu2 = np.mean(nu2_score_element)
        psi_nu2 = nu2_score_element - nu2

        element_dict = {'sigma2': sigma2,
                        'nu2': nu2,
                        'psi_sigma2': psi_sigma2,
                        'psi_nu2': psi_nu2}
        return element_dict

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
        g1_tune_res = _dml_tune(y, x, train_inds_d1,
                                self._learner['ml_g'], param_grids['ml_g'], scoring_methods['ml_g'],
                                n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        m_tune_res = _dml_tune(d, x, train_inds,
                               self._learner['ml_m'], param_grids['ml_m'], scoring_methods['ml_m'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        g0_best_params = [xx.best_params_ for xx in g0_tune_res]
        g1_best_params = [xx.best_params_ for xx in g1_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]

        params = {'ml_g0': g0_best_params,
                  'ml_g1': g1_best_params,
                  'ml_m': m_best_params}
        tune_res = {'g0_tune': g0_tune_res,
                    'g1_tune': g1_tune_res,
                    'm_tune': m_tune_res}

        res = {'params': params,
               'tune_res': tune_res}

        return res

    def cate(self, basis, is_gate=False):
        """
        Calculate conditional average treatment effects (CATE) for a given basis.

        Parameters
        ----------
        basis : :class:`pandas.DataFrame`
            The basis for estimating the best linear predictor. Has to have the shape ``(n_obs, d)``,
            where ``n_obs`` is the number of observations and ``d`` is the number of predictors.
        is_gate : bool
            Indicates whether the basis is constructed for GATEs (dummy-basis).
            Default is ``False``.

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
        model = DoubleMLBLP(orth_signal, basis=basis, is_gate=is_gate)
        model.fit()
        return model

    def gate(self, groups):
        """
        Calculate group average treatment effects (GATE) for groups.

        Parameters
        ----------
        groups : :class:`pandas.DataFrame`
            The group indicator for estimating the best linear predictor. Groups should be mutually exclusive.
            Has to be dummy coded with shape ``(n_obs, d)``, where ``n_obs`` is the number of observations
            and ``d`` is the number of groups or ``(n_obs, 1)`` and contain the corresponding groups (as str).

        Returns
        -------
        model : :class:`doubleML.DoubleMLBLP`
            Best linear Predictor model for Group Effects.
        """
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

        model = self.cate(groups, is_gate=True)
        return model

    def policy_tree(self, features, depth=2, **tree_params):
        """
        Estimate a decision tree for optimal treatment policy by weighted classification.

        Parameters
        ----------
        depth : int
            The depth of the estimated decision tree.
            Has to be larger than 0. Deeper trees derive a more complex decision policy. Default is ``2``.

        features : :class:`pandas.DataFrame`
            The covariates on which the policy tree is learned.
            Has to be of shape ``(n_obs, d)``, where ``n_obs`` is the number of observations
            and ``d`` is the number of covariates to be included.

        **tree_params : dict
            Parameters that are forwarded to the :class:`sklearn.tree.DecisionTreeClassifier`.
            Note that by default we perform minimal pruning by setting the ``ccp_alpha = 0.01`` and
            ``min_samples_leaf = 8``. This can be adjusted.

        Returns
        -------
        model : :class:`doubleML.DoubleMLPolicyTree`
            Policy tree model.
        """
        valid_score = ['ATE']
        if self.score not in valid_score:
            raise ValueError('Invalid score ' + self.score + '. ' +
                             'Valid score ' + ' or '.join(valid_score) + '.')

        if self.n_rep != 1:
            raise NotImplementedError('Only implemented for one repetition. ' +
                                      f'Number of repetitions is {str(self.n_rep)}.')

        _check_integer(depth, "Depth", 0)

        if not isinstance(features, pd.DataFrame):
            raise TypeError('Covariates must be of DataFrame type. '
                            f'Covariates of type {str(type(features))} was passed.')

        orth_signal = self.psi_elements['psi_b'].reshape(-1)

        model = DoubleMLPolicyTree(orth_signal, depth=depth, features=features, **tree_params).fit()

        return model
