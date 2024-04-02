import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor

import warnings

from ..double_ml import DoubleML
from ..double_ml_data import DoubleMLData
from ..double_ml_score_mixins import LinearScoreMixin

from ..utils._estimation import _dml_cv_predict, _dml_tune
from ..utils._checks import _check_finite_predictions


class DoubleMLPLIV(LinearScoreMixin, DoubleML):
    """Double machine learning for partially linear IV regression models

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    ml_l : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`\\ell_0(X) = E[Y|X]`.

    ml_m : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`m_0(X) = E[Z|X]`.

    ml_r : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`r_0(X) = E[D|X]`.

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
        or a callable object / function with signature
        ``psi_a, psi_b = score(y, z, d, l_hat, m_hat, r_hat, g_hat, smpls)``.
        Default is ``'partialling out'``.

    draw_sample_splitting : bool
        Indicates whether the sample splitting should be drawn during initialization of the object.
        Default is ``True``.

    Examples
    --------
    >>> import numpy as np
    >>> import doubleml as dml
    >>> from doubleml.datasets import make_pliv_CHS2015
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.base import clone
    >>> np.random.seed(3141)
    >>> learner = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
    >>> ml_l = clone(learner)
    >>> ml_m = clone(learner)
    >>> ml_r = clone(learner)
    >>> data = make_pliv_CHS2015(alpha=0.5, n_obs=500, dim_x=20, dim_z=1, return_type='DataFrame')
    >>> obj_dml_data = dml.DoubleMLData(data, 'y', 'd', z_cols='Z1')
    >>> dml_pliv_obj = dml.DoubleMLPLIV(obj_dml_data, ml_l, ml_m, ml_r)
    >>> dml_pliv_obj.fit().summary
           coef   std err         t         P>|t|     2.5 %    97.5 %
    d  0.522753  0.082263  6.354688  2.088504e-10  0.361521  0.683984

    Notes
    -----
    **Partially linear IV regression (PLIV)** models take the form

    .. math::

        Y - D \\theta_0 =  g_0(X) + \\zeta, & &\\mathbb{E}(\\zeta | Z, X) = 0,

        Z = m_0(X) + V, & &\\mathbb{E}(V | X) = 0.

    where :math:`Y` is the outcome variable, :math:`D` is the policy variable of interest and :math:`Z`
    denotes one or multiple instrumental variables. The high-dimensional vector
    :math:`X = (X_1, \\ldots, X_p)` consists of other confounding covariates, and :math:`\\zeta` and
    :math:`V` are stochastic errors.
    """

    def __init__(self,
                 obj_dml_data,
                 ml_l,
                 ml_m,
                 ml_r,
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
        self.partialX = True
        self.partialZ = False
        self._check_score(self.score)
        _ = self._check_learner(ml_l, 'ml_l', regressor=True, classifier=False)
        _ = self._check_learner(ml_m, 'ml_m', regressor=True, classifier=False)
        _ = self._check_learner(ml_r, 'ml_r', regressor=True, classifier=False)
        self._learner = {'ml_l': ml_l, 'ml_m': ml_m, 'ml_r': ml_r}
        if ml_g is not None:
            if (isinstance(self.score, str) & (self.score == 'IV-type')) | callable(self.score):
                _ = self._check_learner(ml_g, 'ml_g', regressor=True, classifier=False)
                self._learner['ml_g'] = ml_g
            else:
                assert (isinstance(self.score, str) & (self.score == 'partialling out'))
                warnings.warn(('A learner ml_g has been provided for score = "partialling out" but will be ignored. "'
                               'A learner ml_g is not required for estimation.'))
        elif isinstance(self.score, str) & (self.score == 'IV-type'):
            raise ValueError("For score = 'IV-type', learners ml_l, ml_m, ml_r and ml_g need to be specified.")
        self._predict_method = {'ml_l': 'predict', 'ml_m': 'predict', 'ml_r': 'predict'}
        if 'ml_g' in self._learner:
            self._predict_method['ml_g'] = 'predict'
        self._initialize_ml_nuisance_params()
        self._external_predictions_implemented = True

    @classmethod
    def _partialX(cls,
                  obj_dml_data,
                  ml_l,
                  ml_m,
                  ml_r,
                  ml_g=None,
                  n_folds=5,
                  n_rep=1,
                  score='partialling out',
                  draw_sample_splitting=True):
        obj = cls(obj_dml_data,
                  ml_l,
                  ml_m,
                  ml_r,
                  ml_g,
                  n_folds,
                  n_rep,
                  score,
                  draw_sample_splitting)
        obj._check_data(obj._dml_data)
        obj.partialX = True
        obj.partialZ = False
        obj._check_score(obj.score)
        _ = obj._check_learner(ml_l, 'ml_l', regressor=True, classifier=False)
        _ = obj._check_learner(ml_m, 'ml_m', regressor=True, classifier=False)
        _ = obj._check_learner(ml_r, 'ml_r', regressor=True, classifier=False)
        obj._learner = {'ml_l': ml_l, 'ml_m': ml_m, 'ml_r': ml_r}
        obj._predict_method = {'ml_l': 'predict', 'ml_m': 'predict', 'ml_r': 'predict'}
        obj._initialize_ml_nuisance_params()
        return obj

    @classmethod
    def _partialZ(cls,
                  obj_dml_data,
                  ml_r,
                  n_folds=5,
                  n_rep=1,
                  score='partialling out',
                  draw_sample_splitting=True):
        # to pass the checks for the learners, we temporarily set ml_l and ml_m to DummyRegressor()
        obj = cls(obj_dml_data,
                  DummyRegressor(),
                  DummyRegressor(),
                  ml_r,
                  None,
                  n_folds,
                  n_rep,
                  score,
                  draw_sample_splitting)
        obj._check_data(obj._dml_data)
        obj.partialX = False
        obj.partialZ = True
        obj._check_score(obj.score)
        _ = obj._check_learner(ml_r, 'ml_r', regressor=True, classifier=False)
        obj._learner = {'ml_r': ml_r}
        obj._predict_method = {'ml_r': 'predict'}
        obj._initialize_ml_nuisance_params()
        return obj

    @classmethod
    def _partialXZ(cls,
                   obj_dml_data,
                   ml_l,
                   ml_m,
                   ml_r,
                   n_folds=5,
                   n_rep=1,
                   score='partialling out',
                   draw_sample_splitting=True):
        obj = cls(obj_dml_data,
                  ml_l,
                  ml_m,
                  ml_r,
                  None,
                  n_folds,
                  n_rep,
                  score,
                  draw_sample_splitting)
        obj._check_data(obj._dml_data)
        obj.partialX = True
        obj.partialZ = True
        obj._check_score(obj.score)
        _ = obj._check_learner(ml_l, 'ml_l', regressor=True, classifier=False)
        _ = obj._check_learner(ml_m, 'ml_m', regressor=True, classifier=False)
        _ = obj._check_learner(ml_r, 'ml_r', regressor=True, classifier=False)
        obj._learner = {'ml_l': ml_l, 'ml_m': ml_m, 'ml_r': ml_r}
        obj._predict_method = {'ml_l': 'predict', 'ml_m': 'predict', 'ml_r': 'predict'}
        obj._initialize_ml_nuisance_params()
        return obj

    def _initialize_ml_nuisance_params(self):
        if self.partialX & (not self.partialZ) & (self._dml_data.n_instr > 1):
            param_names = ['ml_l', 'ml_r'] + ['ml_m_' + z_col for z_col in self._dml_data.z_cols]
        else:
            param_names = self._learner.keys()
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols}
                        for learner in param_names}

    def _check_score(self, score):
        if isinstance(score, str):
            if self.partialX & (not self.partialZ) & (self._dml_data.n_instr == 1):
                valid_score = ['partialling out', 'IV-type']
            else:
                valid_score = ['partialling out']
            if score not in valid_score:
                raise ValueError('Invalid score ' + score + '. ' +
                                 'Valid score ' + ' or '.join(valid_score) + '.')
        else:
            if not callable(score):
                raise TypeError('score should be either a string or a callable. '
                                '%r was passed.' % score)
        return score

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError('The data must be of DoubleMLData type. '
                            f'{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed.')
        if obj_dml_data.n_instr == 0:
            raise ValueError('Incompatible data. ' +
                             'At least one variable must be set as instrumental variable. '
                             'To fit a partially linear regression model without instrumental variable(s) '
                             'use DoubleMLPLR instead of DoubleMLPLIV.')
        return

    def _nuisance_est(self, smpls, n_jobs_cv, external_predictions, return_models=False):
        if self.partialX & (not self.partialZ):
            psi_elements, preds = self._nuisance_est_partial_x(smpls, n_jobs_cv, external_predictions, return_models)
        elif (not self.partialX) & self.partialZ:
            psi_elements, preds = self._nuisance_est_partial_z(smpls, n_jobs_cv, return_models)
        else:
            assert (self.partialX & self.partialZ)
            psi_elements, preds = self._nuisance_est_partial_xz(smpls, n_jobs_cv, return_models)

        return psi_elements, preds

    def _nuisance_tuning(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                         search_mode, n_iter_randomized_search):
        if self.partialX & (not self.partialZ):
            res = self._nuisance_tuning_partial_x(smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                                                  search_mode, n_iter_randomized_search)
        elif (not self.partialX) & self.partialZ:
            res = self._nuisance_tuning_partial_z(smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                                                  search_mode, n_iter_randomized_search)
        else:
            assert (self.partialX & self.partialZ)
            res = self._nuisance_tuning_partial_xz(smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                                                   search_mode, n_iter_randomized_search)

        return res

    def _nuisance_est_partial_x(self, smpls, n_jobs_cv, external_predictions, return_models=False):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)

        # nuisance l
        if external_predictions['ml_l'] is not None:
            l_hat = {'preds': external_predictions['ml_l'],
                     'targets': None,
                     'models': None}
        else:
            l_hat = _dml_cv_predict(self._learner['ml_l'], x, y, smpls=smpls, n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_l'), method=self._predict_method['ml_l'],
                                    return_models=return_models)
        _check_finite_predictions(l_hat['preds'], self._learner['ml_l'], 'ml_l', smpls)

        predictions = {'ml_l': l_hat['preds']}
        targets = {'ml_l': l_hat['targets']}
        models = {'ml_l': l_hat['models']}
        # nuisance m
        if self._dml_data.n_instr == 1:
            # one instrument: just identified
            x, z = check_X_y(x, np.ravel(self._dml_data.z),
                             force_all_finite=False)
            if external_predictions['ml_m'] is not None:
                m_hat = {'preds': external_predictions['ml_m'],
                         'targets': None,
                         'models': None}
            else:
                m_hat = _dml_cv_predict(self._learner['ml_m'], x, z, smpls=smpls, n_jobs=n_jobs_cv,
                                        est_params=self._get_params('ml_m'), method=self._predict_method['ml_m'],
                                        return_models=return_models)
            predictions['ml_m'] = m_hat['preds']
            targets['ml_m'] = m_hat['targets']
            models['ml_m'] = m_hat['models']
        else:
            # several instruments: 2SLS
            m_hat = {'preds': np.full((self._dml_data.n_obs, self._dml_data.n_instr), np.nan),
                     'targets': [None] * self._dml_data.n_instr,
                     'models': [None] * self._dml_data.n_instr}
            for i_instr in range(self._dml_data.n_instr):
                z = self._dml_data.z
                x, this_z = check_X_y(x, z[:, i_instr],
                                      force_all_finite=False)
                if external_predictions['ml_m_' + self._dml_data.z_cols[i_instr]] is not None:
                    m_hat['preds'][:, i_instr] = external_predictions['ml_m_' + self._dml_data.z_cols[i_instr]]
                    predictions['ml_m_' + self._dml_data.z_cols[i_instr]] = external_predictions[
                        'ml_m_' + self._dml_data.z_cols[i_instr]]
                    targets['ml_m_' + self._dml_data.z_cols[i_instr]] = None
                    models['ml_m_' + self._dml_data.z_cols[i_instr]] = None
                else:
                    res_cv_predict = _dml_cv_predict(self._learner['ml_m'], x, this_z, smpls=smpls, n_jobs=n_jobs_cv,
                                                     est_params=self._get_params('ml_m_' + self._dml_data.z_cols[i_instr]),
                                                     method=self._predict_method['ml_m'], return_models=return_models)

                    m_hat['preds'][:, i_instr] = res_cv_predict['preds']

                    predictions['ml_m_' + self._dml_data.z_cols[i_instr]] = res_cv_predict['preds']
                    targets['ml_m_' + self._dml_data.z_cols[i_instr]] = res_cv_predict['targets']
                    models['ml_m_' + self._dml_data.z_cols[i_instr]] = res_cv_predict['models']

        _check_finite_predictions(m_hat['preds'], self._learner['ml_m'], 'ml_m', smpls)

        # nuisance r
        if external_predictions['ml_r'] is not None:
            r_hat = {'preds': external_predictions['ml_r'],
                     'targets': None,
                     'models': None}
        else:
            r_hat = _dml_cv_predict(self._learner['ml_r'], x, d, smpls=smpls, n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_r'), method=self._predict_method['ml_r'],
                                    return_models=return_models)
        _check_finite_predictions(r_hat['preds'], self._learner['ml_r'], 'ml_r', smpls)
        predictions['ml_r'] = r_hat['preds']
        targets['ml_r'] = r_hat['targets']
        models['ml_r'] = r_hat['models']

        g_hat = {'preds': None, 'targets': None, 'models': None}
        if (self._dml_data.n_instr == 1) & ('ml_g' in self._learner):
            # an estimate of g is obtained for the IV-type score and callable scores
            # get an initial estimate for theta using the partialling out score
            if external_predictions['ml_g'] is not None:
                g_hat = {'preds': external_predictions['ml_g'],
                         'targets': None,
                         'models': None}
            else:
                psi_a = -np.multiply(d - r_hat['preds'], z - m_hat['preds'])
                psi_b = np.multiply(z - m_hat['preds'], y - l_hat['preds'])
                theta_initial = -np.nanmean(psi_b) / np.nanmean(psi_a)
                # nuisance g
                g_hat = _dml_cv_predict(self._learner['ml_g'], x, y - theta_initial * d, smpls=smpls, n_jobs=n_jobs_cv,
                                        est_params=self._get_params('ml_g'), method=self._predict_method['ml_g'],
                                        return_models=return_models)
            _check_finite_predictions(g_hat['preds'], self._learner['ml_g'], 'ml_g', smpls)

        predictions['ml_g'] = g_hat['preds']
        targets['ml_g'] = g_hat['targets']
        models['ml_g'] = g_hat['models']
        psi_a, psi_b = self._score_elements(y, z, d,
                                            l_hat['preds'], m_hat['preds'], r_hat['preds'], g_hat['preds'],
                                            smpls)
        psi_elements = {'psi_a': psi_a,
                        'psi_b': psi_b}
        predictions = {'predictions': predictions,
                       'targets': targets,
                       'models': models
                       }

        return psi_elements, predictions

    def _score_elements(self, y, z, d, l_hat, m_hat, r_hat, g_hat, smpls):
        # compute residuals
        u_hat = y - l_hat
        w_hat = d - r_hat
        v_hat = z - m_hat

        r_hat_tilde = None
        if self._dml_data.n_instr > 1:
            # projection of w_hat on v_hat
            reg = LinearRegression(fit_intercept=True).fit(v_hat, w_hat)
            r_hat_tilde = reg.predict(v_hat)

        if isinstance(self.score, str):
            if self._dml_data.n_instr == 1:
                if self.score == 'partialling out':
                    psi_a = -np.multiply(w_hat, v_hat)
                    psi_b = np.multiply(v_hat, u_hat)
                else:
                    assert self.score == 'IV-type'
                    psi_a = -np.multiply(v_hat, d)
                    psi_b = np.multiply(v_hat, y - g_hat)
            else:
                assert self.score == 'partialling out'
                psi_a = -np.multiply(w_hat, r_hat_tilde)
                psi_b = np.multiply(r_hat_tilde, u_hat)
        else:
            assert callable(self.score)
            if self._dml_data.n_instr > 1:
                raise NotImplementedError('Callable score not implemented for DoubleMLPLIV.partialX '
                                          'with several instruments.')
            else:
                assert self._dml_data.n_instr == 1
                psi_a, psi_b = self.score(y=y, z=z, d=d,
                                          l_hat=l_hat, m_hat=m_hat, r_hat=r_hat, g_hat=g_hat,
                                          smpls=smpls)

        return psi_a, psi_b

    def _nuisance_est_partial_z(self, smpls, n_jobs_cv, return_models=False):
        y = self._dml_data.y
        xz, d = check_X_y(np.hstack((self._dml_data.x, self._dml_data.z)),
                          self._dml_data.d,
                          force_all_finite=False)

        # nuisance m
        r_hat = _dml_cv_predict(self._learner['ml_r'], xz, d, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_r'), method=self._predict_method['ml_r'],
                                return_models=return_models)
        _check_finite_predictions(r_hat['preds'], self._learner['ml_r'], 'ml_r', smpls)

        if isinstance(self.score, str):
            assert self.score == 'partialling out'
            psi_a = -np.multiply(r_hat['preds'], d)
            psi_b = np.multiply(r_hat['preds'], y)
        else:
            assert callable(self.score)
            raise NotImplementedError('Callable score not implemented for DoubleMLPLIV.partialZ.')

        psi_elements = {'psi_a': psi_a,
                        'psi_b': psi_b}
        preds = {'predictions': {'ml_r': r_hat['preds']},
                 'targets': {'ml_r': r_hat['targets']},
                 'models': {'ml_r': r_hat['models']}}

        return psi_elements, preds

    def _nuisance_est_partial_xz(self, smpls, n_jobs_cv, return_models=False):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        xz, d = check_X_y(np.hstack((self._dml_data.x, self._dml_data.z)),
                          self._dml_data.d,
                          force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)

        # nuisance l
        l_hat = _dml_cv_predict(self._learner['ml_l'], x, y, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_l'), method=self._predict_method['ml_l'],
                                return_models=return_models)
        _check_finite_predictions(l_hat['preds'], self._learner['ml_l'], 'ml_l', smpls)

        # nuisance m
        m_hat = _dml_cv_predict(self._learner['ml_m'], xz, d, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_m'), return_train_preds=True,
                                method=self._predict_method['ml_m'], return_models=return_models)
        _check_finite_predictions(m_hat['preds'], self._learner['ml_m'], 'ml_m', smpls)

        # nuisance r
        m_hat_tilde = _dml_cv_predict(self._learner['ml_r'], x, m_hat['train_preds'], smpls=smpls, n_jobs=n_jobs_cv,
                                      est_params=self._get_params('ml_r'), method=self._predict_method['ml_r'],
                                      return_models=return_models)
        _check_finite_predictions(m_hat_tilde['preds'], self._learner['ml_r'], 'ml_r', smpls)

        # compute residuals
        u_hat = y - l_hat['preds']
        w_hat = d - m_hat_tilde['preds']

        if isinstance(self.score, str):
            assert self.score == 'partialling out'
            psi_a = -np.multiply(w_hat, (m_hat['preds']-m_hat_tilde['preds']))
            psi_b = np.multiply((m_hat['preds']-m_hat_tilde['preds']), u_hat)
        else:
            assert callable(self.score)
            raise NotImplementedError('Callable score not implemented for DoubleMLPLIV.partialXZ.')

        psi_elements = {'psi_a': psi_a,
                        'psi_b': psi_b}
        preds = {'predictions': {'ml_l': l_hat['preds'],
                                 'ml_m': m_hat['preds'],
                                 'ml_r': m_hat_tilde['preds']},
                 'targets': {'ml_l': l_hat['targets'],
                             'ml_m': m_hat['targets'],
                             'ml_r': m_hat_tilde['targets']},
                 'models': {'ml_l': l_hat['models'],
                            'ml_m': m_hat['models'],
                            'ml_r': m_hat_tilde['models']}
                 }

        return psi_elements, preds

    def _nuisance_tuning_partial_x(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                                   search_mode, n_iter_randomized_search):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)

        if scoring_methods is None:
            scoring_methods = {'ml_l': None,
                               'ml_m': None,
                               'ml_r': None,
                               'ml_g': None}

        train_inds = [train_index for (train_index, _) in smpls]
        l_tune_res = _dml_tune(y, x, train_inds,
                               self._learner['ml_l'], param_grids['ml_l'], scoring_methods['ml_l'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        if self._dml_data.n_instr > 1:
            # several instruments: 2SLS
            m_tune_res = {instr_var: list() for instr_var in self._dml_data.z_cols}
            z = self._dml_data.z
            for i_instr in range(self._dml_data.n_instr):
                x, this_z = check_X_y(x, z[:, i_instr],
                                      force_all_finite=False)
                m_tune_res[self._dml_data.z_cols[i_instr]] = _dml_tune(this_z, x, train_inds,
                                                                       self._learner['ml_m'], param_grids['ml_m'],
                                                                       scoring_methods['ml_m'],
                                                                       n_folds_tune, n_jobs_cv, search_mode,
                                                                       n_iter_randomized_search)
        else:
            # one instrument: just identified
            x, z = check_X_y(x, np.ravel(self._dml_data.z),
                             force_all_finite=False)
            m_tune_res = _dml_tune(z, x, train_inds,
                                   self._learner['ml_m'], param_grids['ml_m'], scoring_methods['ml_m'],
                                   n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        r_tune_res = _dml_tune(d, x, train_inds,
                               self._learner['ml_r'], param_grids['ml_r'], scoring_methods['ml_r'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        l_best_params = [xx.best_params_ for xx in l_tune_res]
        r_best_params = [xx.best_params_ for xx in r_tune_res]
        if self._dml_data.n_instr > 1:
            params = {'ml_l': l_best_params,
                      'ml_r': r_best_params}
            for instr_var in self._dml_data.z_cols:
                params['ml_m_' + instr_var] = [xx.best_params_ for xx in m_tune_res[instr_var]]
            tune_res = {'l_tune': l_tune_res,
                        'm_tune': m_tune_res,
                        'r_tune': r_tune_res}
        else:
            m_best_params = [xx.best_params_ for xx in m_tune_res]
            # an ML model for g is obtained for the IV-type score and callable scores
            if 'ml_g' in self._learner:
                # construct an initial theta estimate from the tuned models using the partialling out score
                l_hat = np.full_like(y, np.nan)
                m_hat = np.full_like(z, np.nan)
                r_hat = np.full_like(d, np.nan)
                for idx, (train_index, _) in enumerate(smpls):
                    l_hat[train_index] = l_tune_res[idx].predict(x[train_index, :])
                    m_hat[train_index] = m_tune_res[idx].predict(x[train_index, :])
                    r_hat[train_index] = r_tune_res[idx].predict(x[train_index, :])
                psi_a = -np.multiply(d - r_hat, z - m_hat)
                psi_b = np.multiply(z - m_hat, y - l_hat)
                theta_initial = -np.nanmean(psi_b) / np.nanmean(psi_a)
                g_tune_res = _dml_tune(y - theta_initial * d, x, train_inds,
                                       self._learner['ml_g'], param_grids['ml_g'], scoring_methods['ml_g'],
                                       n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)
                g_best_params = [xx.best_params_ for xx in g_tune_res]

                params = {'ml_l': l_best_params,
                          'ml_m': m_best_params,
                          'ml_r': r_best_params,
                          'ml_g': g_best_params}
                tune_res = {'l_tune': l_tune_res,
                            'm_tune': m_tune_res,
                            'r_tune': r_tune_res,
                            'g_tune': g_tune_res}
            else:
                params = {'ml_l': l_best_params,
                          'ml_m': m_best_params,
                          'ml_r': r_best_params}
                tune_res = {'l_tune': l_tune_res,
                            'm_tune': m_tune_res,
                            'r_tune': r_tune_res}

        res = {'params': params,
               'tune_res': tune_res}

        return res

    def _nuisance_tuning_partial_z(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                                   search_mode, n_iter_randomized_search):
        xz, d = check_X_y(np.hstack((self._dml_data.x, self._dml_data.z)),
                          self._dml_data.d,
                          force_all_finite=False)

        if scoring_methods is None:
            scoring_methods = {'ml_r': None}

        train_inds = [train_index for (train_index, _) in smpls]
        m_tune_res = _dml_tune(d, xz, train_inds,
                               self._learner['ml_r'], param_grids['ml_r'], scoring_methods['ml_r'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        m_best_params = [xx.best_params_ for xx in m_tune_res]

        params = {'ml_r': m_best_params}

        tune_res = {'r_tune': m_tune_res}

        res = {'params': params,
               'tune_res': tune_res}

        return res

    def _nuisance_tuning_partial_xz(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                                    search_mode, n_iter_randomized_search):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        xz, d = check_X_y(np.hstack((self._dml_data.x, self._dml_data.z)),
                          self._dml_data.d,
                          force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)

        if scoring_methods is None:
            scoring_methods = {'ml_l': None,
                               'ml_m': None,
                               'ml_r': None}

        train_inds = [train_index for (train_index, _) in smpls]
        l_tune_res = _dml_tune(y, x, train_inds,
                               self._learner['ml_l'], param_grids['ml_l'], scoring_methods['ml_l'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)
        m_tune_res = _dml_tune(d, xz, train_inds,
                               self._learner['ml_m'], param_grids['ml_m'], scoring_methods['ml_m'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        r_tune_res = list()
        for idx, (train_index, _) in enumerate(smpls):
            m_hat = m_tune_res[idx].predict(xz[train_index, :])
            r_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
            if search_mode == 'grid_search':
                r_grid_search = GridSearchCV(self._learner['ml_r'], param_grids['ml_r'],
                                             scoring=scoring_methods['ml_r'],
                                             cv=r_tune_resampling, n_jobs=n_jobs_cv)
            else:
                assert search_mode == 'randomized_search'
                r_grid_search = RandomizedSearchCV(self._learner['ml_r'], param_grids['ml_r'],
                                                   scoring=scoring_methods['ml_r'],
                                                   cv=r_tune_resampling, n_jobs=n_jobs_cv,
                                                   n_iter=n_iter_randomized_search)
            r_tune_res.append(r_grid_search.fit(x[train_index, :], m_hat))

        l_best_params = [xx.best_params_ for xx in l_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]
        r_best_params = [xx.best_params_ for xx in r_tune_res]

        params = {'ml_l': l_best_params,
                  'ml_m': m_best_params,
                  'ml_r': r_best_params}

        tune_res = {'l_tune': l_tune_res,
                    'm_tune': m_tune_res,
                    'r_tune': r_tune_res}

        res = {'params': params,
               'tune_res': tune_res}

        return res

    def _sensitivity_element_est(self, preds):
        pass
