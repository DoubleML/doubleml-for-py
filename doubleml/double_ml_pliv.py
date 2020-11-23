import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor

from .double_ml import DoubleML
from ._helper import _dml_cv_predict


class DoubleMLPLIV(DoubleML):
    """
    Double machine learning for partially linear IV regression models

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    ml_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g. :py:class:`sklearn.ensemble.RandomForestRegressor`)
        for the nuisance function :math:`g_0(X) = E[Y|X]`.

    ml_m : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g. :py:class:`sklearn.ensemble.RandomForestRegressor`)
        for the nuisance function :math:`m_0(X) = E[Z|X]`.

    ml_r : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g. :py:class:`sklearn.ensemble.RandomForestRegressor`)
        for the nuisance function :math:`r_0(X) = E[D|X]`.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitons for the sample splitting.
        Default is ``1``.

    score : str or callable
        A str (``'partialling out'`` is the only choice) specifying the score function
        or a callable object / function with signature ``psi_a, psi_b = score(y, z, d, g_hat, m_hat, r_hat, smpls)``.
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
    >>> from doubleml.datasets import make_pliv_CHS2015
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.base import clone
    >>> np.random.seed(3141)
    >>> learner = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
    >>> ml_g = clone(learner)
    >>> ml_m = clone(learner)
    >>> ml_r = clone(learner)
    >>> data = make_pliv_CHS2015(alpha=0.5, n_obs=500, dim_x=20, dim_z=1, return_type='DataFrame')
    >>> obj_dml_data = dml.DoubleMLData(data, 'y', 'd', z_cols='Z1')
    >>> dml_pliv_obj = dml.DoubleMLPLIV(obj_dml_data, ml_g, ml_m, ml_r)
    >>> dml_pliv_obj.fit().summary
           coef   std err         t         P>|t|     2.5 %    97.5 %
    d  0.522753  0.082263  6.354688  2.088504e-10  0.361521  0.683984

    Notes
    -----
    .. include:: ../../shared/models/pliv.rst
    """
    def __init__(self,
                 obj_dml_data,
                 ml_g,
                 ml_m,
                 ml_r,
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
        self.partialX = True
        self.partialZ = False
        self._learner = {'ml_g': self._check_learner(ml_g, 'ml_g'),
                         'ml_m': self._check_learner(ml_m, 'ml_m'),
                         'ml_r': self._check_learner(ml_r, 'ml_r')}
        self._initialize_ml_nuisance_params()

    @classmethod
    def _partialX(cls,
                  obj_dml_data,
                  ml_g,
                  ml_m,
                  ml_r,
                  n_folds=5,
                  n_rep=1,
                  score='partialling out',
                  dml_procedure='dml2',
                  draw_sample_splitting=True,
                  apply_cross_fitting=True):
        obj = cls(obj_dml_data,
                  ml_g,
                  ml_m,
                  ml_r,
                  n_folds,
                  n_rep,
                  score,
                  dml_procedure,
                  draw_sample_splitting,
                  apply_cross_fitting)
        obj.partialX = True
        obj.partialZ = False
        obj._learner = {'ml_g': obj._check_learner(ml_g, 'ml_g'),
                        'ml_m': obj._check_learner(ml_m, 'ml_m'),
                        'ml_r': obj._check_learner(ml_r, 'ml_r')}
        obj._initialize_ml_nuisance_params()
        return obj

    @classmethod
    def _partialZ(cls,
                  obj_dml_data,
                  ml_r,
                  n_folds=5,
                  n_rep=1,
                  score='partialling out',
                  dml_procedure='dml2',
                  draw_sample_splitting=True,
                  apply_cross_fitting=True):
        # to pass the checks for the learners, we temporarily set ml_g and ml_m to DummyRegressor()
        obj = cls(obj_dml_data,
                  DummyRegressor(),
                  DummyRegressor(),
                  ml_r,
                  n_folds,
                  n_rep,
                  score,
                  dml_procedure,
                  draw_sample_splitting,
                  apply_cross_fitting)
        obj.partialX = False
        obj.partialZ = True
        obj._learner = {'ml_r': obj._check_learner(ml_r, 'ml_r')}
        obj._initialize_ml_nuisance_params()
        return obj

    @classmethod
    def _partialXZ(cls,
                   obj_dml_data,
                   ml_g,
                   ml_m,
                   ml_r,
                   n_folds=5,
                   n_rep=1,
                   score='partialling out',
                   dml_procedure='dml2',
                   draw_sample_splitting=True,
                   apply_cross_fitting=True):
        obj = cls(obj_dml_data,
                  ml_g,
                  ml_m,
                  ml_r,
                  n_folds,
                  n_rep,
                  score,
                  dml_procedure,
                  draw_sample_splitting,
                  apply_cross_fitting)
        obj.partialX = True
        obj.partialZ = True
        obj._learner = {'ml_g': obj._check_learner(ml_g, 'ml_g'),
                        'ml_m': obj._check_learner(ml_m, 'ml_m'),
                        'ml_r': obj._check_learner(ml_r, 'ml_r')}
        obj._initialize_ml_nuisance_params()
        return obj

    def _initialize_ml_nuisance_params(self):
        if self.partialX & (not self.partialZ):
            if self._dml_data.n_instr == 1:
                valid_learner = ['ml_g', 'ml_m', 'ml_r']
            else:
                valid_learner = ['ml_g', 'ml_r'] + ['ml_m_' + z_col for z_col in self._dml_data.z_cols]
        elif (not self.partialX) & self.partialZ:
            valid_learner = ['ml_r']
        else:
            assert (self.partialX & self.partialZ)
            valid_learner = ['ml_g', 'ml_m', 'ml_r']
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols}
                        for learner in valid_learner}

    def _check_score(self, score):
        if isinstance(score, str):
            valid_score = ['partialling out']
            # check whether its worth implementing the IV_type as well
            # In CCDHNR equation (4.7) a score of this type is provided;
            # however in the following paragraph it is explained that one might
            # still need to estimate the partialling out type first
            if score not in valid_score:
                raise ValueError('invalid score ' + score +
                                 '\n valid score ' + 'partialling out')
        else:
            if not callable(score):
                raise ValueError('score should be either a string or a callable.'
                                 ' %r was passed' % score)
        return score

    def _check_data(self, obj_dml_data):
        return

    def _ml_nuisance_and_score_elements(self, smpls, n_jobs_cv):
        if self.partialX & (not self.partialZ):
            psi_a, psi_b = self._ml_nuisance_and_score_elements_partial_x(smpls, n_jobs_cv)
        elif (not self.partialX) & self.partialZ:
            psi_a, psi_b = self._ml_nuisance_and_score_elements_partial_z(smpls, n_jobs_cv)
        else:
            assert (self.partialX & self.partialZ)
            psi_a, psi_b = self._ml_nuisance_and_score_elements_partial_xz(smpls, n_jobs_cv)

        return psi_a, psi_b

    def _ml_nuisance_tuning(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                            search_mode, n_iter_randomized_search):
        if self.partialX & (not self.partialZ):
            res = self._ml_nuisance_tuning_partial_x(smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                                                     search_mode, n_iter_randomized_search)
        elif (not self.partialX) & self.partialZ:
            res = self._ml_nuisance_tuning_partial_z(smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                                                     search_mode, n_iter_randomized_search)
        else:
            assert (self.partialX & self.partialZ)
            res = self._ml_nuisance_tuning_partial_xz(smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                                                      search_mode, n_iter_randomized_search)

        return res

    def _ml_nuisance_and_score_elements_partial_x(self, smpls, n_jobs_cv):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y)
        x, d = check_X_y(x, self._dml_data.d)
        
        # nuisance g
        g_hat = _dml_cv_predict(self._learner['ml_g'], x, y, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_g'))
        
        # nuisance m
        if self._dml_data.n_instr == 1:
            # one instrument: just identified
            x, z = check_X_y(x, np.ravel(self._dml_data.z))
            m_hat = _dml_cv_predict(self._learner['ml_m'], x, z, smpls=smpls, n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_m'))
        else:
            # several instruments: 2SLS
            m_hat = np.full((self._dml_data.n_obs, self._dml_data.n_instr), np.nan)
            z = self._dml_data.z
            for i_instr in range(self._dml_data.n_instr):
                x, this_z = check_X_y(x, z[:, i_instr])
                m_hat[:, i_instr] = _dml_cv_predict(self._learner['ml_m'], x, this_z, smpls=smpls, n_jobs=n_jobs_cv,
                                                    est_params=self._get_params('ml_m_' + self._dml_data.z_cols[i_instr]))

        # nuisance r
        r_hat = _dml_cv_predict(self._learner['ml_r'], x, d, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_r'))
        
        # compute residuals
        u_hat = y - g_hat
        w_hat = d - r_hat
        v_hat = z - m_hat

        r_hat_tilde = None
        if self._dml_data.n_instr > 1:
            assert self.apply_cross_fitting
            # TODO check whether the no cross-fitting case can be supported here
            # projection of w_hat on v_hat
            reg = LinearRegression(fit_intercept=True).fit(v_hat, w_hat)
            r_hat_tilde = reg.predict(v_hat)

        if isinstance(self.score, str):
            assert self.score == 'partialling out'
            if self._dml_data.n_instr == 1:
                psi_a = -np.multiply(w_hat, v_hat)
                psi_b = np.multiply(v_hat, u_hat)
            else:
                psi_a = -np.multiply(w_hat, r_hat_tilde)
                psi_b = np.multiply(r_hat_tilde, u_hat)
        else:
            assert callable(self.score)
            if self._dml_data.n_instr > 1:
                raise NotImplementedError('callable score not implemented for DoubleMLPLIV.partialX '
                                          'with several instruments')
            else:
                assert self._dml_data.n_instr == 1
                psi_a, psi_b = self.score(y, z, d,
                                          g_hat, m_hat, r_hat, smpls)

        return psi_a, psi_b

    def _ml_nuisance_and_score_elements_partial_z(self, smpls, n_jobs_cv):
        y = self._dml_data.y
        xz, d = check_X_y(np.hstack((self._dml_data.x, self._dml_data.z)),
                          self._dml_data.d)

        # nuisance m
        r_hat = _dml_cv_predict(self._learner['ml_r'], xz, d, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_r'))

        if isinstance(self.score, str):
            assert self.score == 'partialling out'
            psi_a = -np.multiply(r_hat, d)
            psi_b = np.multiply(r_hat, y)
        else:
            assert callable(self.score)
            raise NotImplementedError('callable score not implemented for DoubleMLPLIV.partialZ')

        return psi_a, psi_b

    def _ml_nuisance_and_score_elements_partial_xz(self, smpls, n_jobs_cv):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y)
        xz, d = check_X_y(np.hstack((self._dml_data.x, self._dml_data.z)),
                          self._dml_data.d)
        x, d = check_X_y(x, self._dml_data.d)

        # nuisance g
        g_hat = _dml_cv_predict(self._learner['ml_g'], x, y, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_g'))

        # nuisance m
        m_hat, m_hat_on_train = _dml_cv_predict(self._learner['ml_m'], xz, d, smpls=smpls, n_jobs=n_jobs_cv,
                                                est_params=self._get_params('ml_m'), return_train_preds=True)

        # nuisance r
        m_hat_tilde = _dml_cv_predict(self._learner['ml_r'], x, m_hat_on_train, smpls=smpls, n_jobs=n_jobs_cv,
                                      est_params=self._get_params('ml_r'))

        # compute residuals
        u_hat = y - g_hat
        w_hat = d - m_hat_tilde

        if isinstance(self.score, str):
            assert self.score == 'partialling out'
            psi_a = -np.multiply(w_hat, (m_hat-m_hat_tilde))
            psi_b = np.multiply((m_hat-m_hat_tilde), u_hat)
        else:
            assert callable(self.score)
            raise NotImplementedError('callable score not implemented for DoubleMLPLIV.partialXZ')

        return psi_a, psi_b

    def _ml_nuisance_tuning_partial_x(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                                      search_mode, n_iter_randomized_search):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y)
        x, d = check_X_y(x, self._dml_data.d)

        if scoring_methods is None:
            scoring_methods = {'ml_g': None,
                               'ml_m': None,
                               'ml_r': None}

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

        if self._dml_data.n_instr > 1:
            # several instruments: 2SLS
            m_tune_res = {instr_var: list() for instr_var in self._dml_data.z_cols}
            z = self._dml_data.z
            for i_instr in range(self._dml_data.n_instr):
                for idx, (train_index, test_index) in enumerate(smpls):
                    x, this_z = check_X_y(x, z[:, i_instr])
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
                    m_tune_res[self._dml_data.z_cols[i_instr]].append(m_grid_search.fit(x[train_index, :],
                                                                                        this_z[train_index]))
        else:
            # one instrument: just identified
            m_tune_res = list()
            for idx, (train_index, test_index) in enumerate(smpls):
                x, z = check_X_y(x, np.ravel(self._dml_data.z))
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
                m_tune_res.append(m_grid_search.fit(x[train_index, :], z[train_index]))

        r_tune_res = list()
        for idx, (train_index, test_index) in enumerate(smpls):
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
            r_tune_res.append(r_grid_search.fit(x[train_index, :], d[train_index]))

        g_best_params = [xx.best_params_ for xx in g_tune_res]
        r_best_params = [xx.best_params_ for xx in r_tune_res]
        if self._dml_data.n_instr > 1:
            params = {'ml_g': g_best_params,
                      'ml_r': r_best_params}
            for instr_var in self._dml_data.z_cols:
                params['ml_m_' + instr_var] = [xx.best_params_ for xx in m_tune_res[instr_var]]
        else:
            m_best_params = [xx.best_params_ for xx in m_tune_res]
            params = {'ml_g': g_best_params,
                      'ml_m': m_best_params,
                      'ml_r': r_best_params}

        tune_res = {'g_tune': g_tune_res,
                    'm_tune': m_tune_res,
                    'r_tune': r_tune_res}

        res = {'params': params,
               'tune_res': tune_res}

        return res

    def _ml_nuisance_tuning_partial_z(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                                      search_mode, n_iter_randomized_search):
        xz, d = check_X_y(np.hstack((self._dml_data.x, self._dml_data.z)),
                          self._dml_data.d)

        if scoring_methods is None:
            scoring_methods = {'ml_r': None}

        m_tune_res = list()
        for idx, (train_index, test_index) in enumerate(smpls):
            m_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
            if search_mode == 'grid_search':
                m_grid_search = GridSearchCV(self._learner['ml_r'], param_grids['ml_r'],
                                             scoring=scoring_methods['ml_r'],
                                             cv=m_tune_resampling, n_jobs=n_jobs_cv)
            else:
                assert search_mode == 'randomized_search'
                m_grid_search = RandomizedSearchCV(self._learner['ml_r'], param_grids['ml_r'],
                                                   scoring=scoring_methods['ml_r'],
                                                   cv=m_tune_resampling, n_jobs=n_jobs_cv,
                                                   n_iter=n_iter_randomized_search)
            m_tune_res.append(m_grid_search.fit(xz[train_index, :], d[train_index]))

        m_best_params = [xx.best_params_ for xx in m_tune_res]

        params = {'ml_r': m_best_params}

        tune_res = {'r_tune': m_tune_res}

        res = {'params': params,
               'tune_res': tune_res}

        return res

    def _ml_nuisance_tuning_partial_xz(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                                       search_mode, n_iter_randomized_search):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y)
        xz, d = check_X_y(np.hstack((self._dml_data.x, self._dml_data.z)),
                          self._dml_data.d)
        x, d = check_X_y(x, self._dml_data.d)

        if scoring_methods is None:
            scoring_methods = {'ml_g': None,
                               'ml_m': None,
                               'ml_r': None}

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
            m_tune_res.append(m_grid_search.fit(xz[train_index, :], d[train_index]))

        r_tune_res = list()
        for idx, (train_index, test_index) in enumerate(smpls):
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

        g_best_params = [xx.best_params_ for xx in g_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]
        r_best_params = [xx.best_params_ for xx in r_tune_res]

        params = {'ml_g': g_best_params,
                  'ml_m': m_best_params,
                  'ml_r': r_best_params}

        tune_res = {'g_tune': g_tune_res,
                    'm_tune': m_tune_res,
                    'r_tune': r_tune_res}

        res = {'params': params,
               'tune_res': tune_res}

        return res
