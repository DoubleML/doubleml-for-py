import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from .double_ml import DoubleML
from ._helper import _dml_cv_predict


class DoubleMLPLR(DoubleML):
    """
    Double machine learning for partially linear regression models

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    ml_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g. :py:class:`sklearn.ensemble.RandomForestRegressor`)
        for the nuisance function :math:`g_0(X) = E[Y|X]`.

    ml_m : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g. :py:class:`sklearn.ensemble.RandomForestRegressor`)
        for the nuisance function :math:`m_0(X) = E[D|X]`.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitons for the sample splitting.
        Default is ``1``.

    score : str or callable
        A str (``'partialling out'`` or ``'IV-type'``) specifying the score function
        or a callable object / function with signature ``psi_a, psi_b = score(y, d, g_hat, m_hat, smpls)``.
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
    .. include:: ../../shared/models/plr.rst
    """
    def __init__(self,
                 obj_dml_data,
                 ml_g,
                 ml_m,
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
        self._learner = {'ml_g': self._check_learner(ml_g, 'ml_g'),
                         'ml_m': self._check_learner(ml_m, 'ml_m')}
        self._initialize_ml_nuisance_params()

    def _initialize_ml_nuisance_params(self):
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in ['ml_g', 'ml_m']}

    def _check_score(self, score):
        if isinstance(score, str):
            valid_score = ['IV-type', 'partialling out']
            if score not in valid_score:
                raise ValueError('invalid score ' + score +
                                 '\n valid score ' + ' or '.join(valid_score))
        else:
            if not callable(score):
                raise ValueError('score should be either a string or a callable.'
                                 ' %r was passed' % score)
        return score

    def _check_data(self, obj_dml_data):
        if obj_dml_data.z_cols is not None:
            raise ValueError('Incompatible data.\n'
                             ' and '.join(obj_dml_data.z_cols) +
                             'have been set as instrumental variable(s).\n'
                             'To fit an partially linear IV regression model use DoubleMLPLIV instead of DoubleMLPLR.')
        return

    def _ml_nuisance_and_score_elements(self, smpls, n_jobs_cv):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y)
        x, d = check_X_y(x, self._dml_data.d)
        
        # nuisance g
        g_hat = _dml_cv_predict(self._learner['ml_g'], x, y, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_g'))
        
        # nuisance m
        m_hat = _dml_cv_predict(self._learner['ml_m'], x, d, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_m'))

        # compute residuals
        u_hat = y - g_hat
        v_hat = d - m_hat
        v_hatd = np.multiply(v_hat, d)

        if isinstance(self.score, str):
            if self.score == 'IV-type':
                psi_a = -v_hatd
            else:
                assert self.score == 'partialling out'
                psi_a = -np.multiply(v_hat, v_hat)
            psi_b = np.multiply(v_hat, u_hat)
        else:
            assert callable(self.score)
            psi_a, psi_b = self.score(y, d, g_hat, m_hat, smpls)
        
        return psi_a, psi_b

    def _ml_nuisance_tuning(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                            search_mode, n_iter_randomized_search):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y)
        x, d = check_X_y(x, self._dml_data.d)

        if scoring_methods is None:
            scoring_methods = {'ml_g': None,
                               'ml_m': None}

        g_tune_res = list()
        for idx, (train_index, test_index) in enumerate(smpls):
            g_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
            if search_mode == 'grid_search':
                g_grid_search = GridSearchCV(self._learner['ml_g'], param_grids['ml_g'],
                                             scoring=scoring_methods['ml_g'],
                                             cv=g_tune_resampling, n_jobs=n_jobs_cv)
            else:
                assert search_mode == 'randomized_search'
                g_grid_search = RandomizedSearchCV(self._learner['ml_g'], param_grids['ml_g'],
                                                   scoring=scoring_methods['ml_g'],
                                                   cv=g_tune_resampling, n_jobs=n_jobs_cv,
                                                   n_iter=n_iter_randomized_search)
            g_tune_res.append(g_grid_search.fit(x[train_index, :], y[train_index]))

        m_tune_res = list()
        for idx, (train_index, test_index) in enumerate(smpls):
            m_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
            if search_mode == 'grid_search':
                m_grid_search = GridSearchCV(self._learner['ml_m'], param_grids['ml_m'],
                                             scoring=scoring_methods['ml_m'],
                                             cv=m_tune_resampling, n_jobs=n_jobs_cv)
            else:
                assert search_mode == 'randomized_search'
                m_grid_search = RandomizedSearchCV(self._learner['ml_m'], param_grids['ml_m'],
                                                   scoring=scoring_methods['ml_m'],
                                                   cv=m_tune_resampling, n_jobs=n_jobs_cv,
                                                   n_iter=n_iter_randomized_search)
            m_tune_res.append(m_grid_search.fit(x[train_index, :], d[train_index]))

        g_best_params = [xx.best_params_ for xx in g_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]

        params = {'ml_g': g_best_params,
                  'ml_m': m_best_params}

        tune_res = {'g_tune': g_tune_res,
                    'm_tune': m_tune_res}

        res = {'params': params,
               'tune_res': tune_res}

        return res
