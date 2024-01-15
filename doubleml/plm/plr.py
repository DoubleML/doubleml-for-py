import numpy as np
import pandas as pd
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target
from sklearn.base import clone

import warnings

from ..double_ml import DoubleML
from ..double_ml_data import DoubleMLData
from ..double_ml_score_mixins import LinearScoreMixin
from ..utils.blp import DoubleMLBLP

from ..utils._estimation import _dml_cv_predict, _dml_tune
from ..utils._checks import _check_score, _check_finite_predictions, _check_is_propensity


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
                 ml_l,
                 ml_m,
                 ml_g=None,
                 n_folds=5,
                 n_rep=1,
                 score='partialling out',
                 draw_sample_splitting=True):
        super().__init__(obj_dml_data,
                         n_folds,
                         n_rep,
                         score,
                         draw_sample_splitting)

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
        self._sensitivity_implemented = True
        self._external_predictions_implemented = True

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

    def _nuisance_est(self, smpls, n_jobs_cv, external_predictions, return_models=False):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)
        m_external = external_predictions['ml_m'] is not None
        l_external = external_predictions['ml_l'] is not None
        if 'ml_g' in self._learner:
            g_external = external_predictions['ml_g'] is not None
        else:
            g_external = False

        # nuisance l
        if l_external:
            l_hat = {'preds': external_predictions['ml_l'],
                     'targets': None,
                     'models': None}
        elif self._score == "IV-type" and g_external:
            l_hat = {'preds': None,
                     'targets': None,
                     'models': None}
        else:
            l_hat = _dml_cv_predict(self._learner['ml_l'], x, y, smpls=smpls, n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_l'), method=self._predict_method['ml_l'],
                                    return_models=return_models)
            _check_finite_predictions(l_hat['preds'], self._learner['ml_l'], 'ml_l', smpls)

        # nuisance m
        if m_external:
            m_hat = {'preds': external_predictions['ml_m'],
                     'targets': None,
                     'models': None}
        else:
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
            # nuisance g
            if g_external:
                g_hat = {'preds': external_predictions['ml_g'],
                         'targets': None,
                         'models': None}
            else:
                # get an initial estimate for theta using the partialling out score
                psi_a = -np.multiply(d - m_hat['preds'], d - m_hat['preds'])
                psi_b = np.multiply(d - m_hat['preds'], y - l_hat['preds'])
                theta_initial = -np.nanmean(psi_b) / np.nanmean(psi_a)
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
        # compute residual
        v_hat = d - m_hat

        if isinstance(self.score, str):
            if self.score == 'IV-type':
                psi_a = - np.multiply(v_hat, d)
                psi_b = np.multiply(v_hat, y - g_hat)
            else:
                assert self.score == 'partialling out'
                u_hat = y - l_hat
                psi_a = -np.multiply(v_hat, v_hat)
                psi_b = np.multiply(v_hat, u_hat)
        else:
            assert callable(self.score)
            psi_a, psi_b = self.score(y=y, d=d,
                                      l_hat=l_hat, m_hat=m_hat, g_hat=g_hat,
                                      smpls=smpls)

        return psi_a, psi_b

    def _sensitivity_element_est(self, preds):
        # set elments for readability
        y = self._dml_data.y
        d = self._dml_data.d

        m_hat = preds['predictions']['ml_m']
        theta = self.all_coef[self._i_treat, self._i_rep]

        if self.score == 'partialling out':
            l_hat = preds['predictions']['ml_l']
            sigma2_score_element = np.square(y - l_hat - np.multiply(theta, d-m_hat))
        else:
            assert self.score == 'IV-type'
            g_hat = preds['predictions']['ml_g']
            sigma2_score_element = np.square(y - g_hat - np.multiply(theta, d))

        sigma2 = np.mean(sigma2_score_element)
        psi_sigma2 = sigma2_score_element - sigma2

        nu2 = np.divide(1.0, np.mean(np.square(d - m_hat)))
        psi_nu2 = nu2 - np.multiply(np.square(d-m_hat), np.square(nu2))

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
        if self._dml_data.n_treat > 1:
            raise NotImplementedError('Only implemented for single treatment. ' +
                                      f'Number of treatments is {str(self._dml_data.n_treat)}.')
        if self.n_rep != 1:
            raise NotImplementedError('Only implemented for one repetition. ' +
                                      f'Number of repetitions is {str(self.n_rep)}.')

        Y_tilde, D_tilde = self._partial_out()

        D_basis = basis * D_tilde
        model = DoubleMLBLP(
            orth_signal=Y_tilde.reshape(-1),
            basis=D_basis,
            is_gate=is_gate,
        )
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

    def _partial_out(self):
        """
        Helper function. Returns the partialled out quantities of Y and D.
        Works with multiple repetitions.

        Returns
        -------
        Y_tilde : :class:`numpy.ndarray`
            The residual of the regression of Y on X.
        D_tilde : :class:`numpy.ndarray`
            The residual of the regression of D on X.
        """
        if self.predictions is None:
            raise ValueError('predictions are None. Call .fit(store_predictions=True) to store the predictions.')

        y = self._dml_data.y.reshape(-1, 1)
        d = self._dml_data.d.reshape(-1, 1)
        ml_m = self.predictions["ml_m"].squeeze(axis=2)

        if self.score == "partialling out":
            ml_l = self.predictions["ml_l"].squeeze(axis=2)
            Y_tilde = y - ml_l
            D_tilde = d - ml_m
        else:
            assert self.score == "IV-type"
            ml_g = self.predictions["ml_g"].squeeze(axis=2)
            Y_tilde = y - (self.coef * ml_m) - ml_g
            D_tilde = d - ml_m

        return Y_tilde, D_tilde
