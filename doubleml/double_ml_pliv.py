import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

from .double_ml import DoubleML, DoubleMLData
from ._helper import _dml_cv_predict


class DoubleMLPLIV(DoubleML):
    """
    Double machine learning for partially linear IV regression models

    Parameters
    ----------
    obj_dml_data :
        ToDo
    ml_learners :
        ToDo
    n_folds :
        ToDo
    n_rep_cross_fit :
        ToDo
    score :
        ToDo
    dml_procedure :
        ToDo
    draw_sample_splitting :
        ToDo
    apply_cross_fitting :
        ToDo

    Examples
    --------
    >>> import doubleml as dml
    >>> from doubleml.datasets import make_pliv_data
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.base import clone
    >>> learner = RandomForestRegressor(max_depth=2, n_estimators=10)
    >>> ml_g = clone(learner)
    >>> ml_m = clone(learner)
    >>> ml_r = clone(learner)
    >>> data = make_pliv_data()
    >>> obj_dml_data = dml.DoubleMLData(data, 'y', 'd', z_cols='z')
    >>> dml_pliv_obj = dml.DoubleMLPLIV(obj_dml_data, ml_g, ml_m, ml_r)
    >>> dml_pliv_obj.fit()
    >>> dml_pliv_obj.summary

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
                 n_rep_cross_fit=1,
                 score='partialling out',
                 dml_procedure='dml2',
                 draw_sample_splitting=True,
                 apply_cross_fitting=True):
        super().__init__(obj_dml_data,
                         n_folds,
                         n_rep_cross_fit,
                         score,
                         dml_procedure,
                         draw_sample_splitting,
                         apply_cross_fitting)
        self.partialX = True
        self.partialZ = False
        self.ml_g = ml_g
        self.ml_m = ml_m
        self.ml_r = ml_r
        self._initialize_ml_nuisance_params()

    @classmethod
    def partialX(cls,
                 obj_dml_data,
                 ml_g,
                 ml_m,
                 ml_r,
                 n_folds=5,
                 n_rep_cross_fit=1,
                 score='partialling out',
                 dml_procedure='dml2',
                 draw_sample_splitting=True,
                 apply_cross_fitting=True):
        obj = cls(obj_dml_data,
                  ml_g,
                  ml_m,
                  ml_r,
                  n_folds,
                  n_rep_cross_fit,
                  score,
                  dml_procedure,
                  draw_sample_splitting,
                  apply_cross_fitting)
        obj._initialize_ml_nuisance_params()
        return obj

    @classmethod
    def partialZ(cls,
                 obj_dml_data,
                 ml_r,
                 n_folds=5,
                 n_rep_cross_fit=1,
                 score='partialling out',
                 dml_procedure='dml2',
                 draw_sample_splitting=True,
                 apply_cross_fitting=True):
        obj = cls(obj_dml_data,
                  None,
                  None,
                  ml_r,
                  n_folds,
                  n_rep_cross_fit,
                  score,
                  dml_procedure,
                  draw_sample_splitting,
                  apply_cross_fitting)
        obj.partialX = False
        obj.partialZ = True
        obj._initialize_ml_nuisance_params()
        return obj

    @classmethod
    def partialXZ(cls,
                  obj_dml_data,
                  ml_g,
                  ml_m,
                  ml_r,
                  n_folds=5,
                  n_rep_cross_fit=1,
                  score='partialling out',
                  dml_procedure='dml2',
                  draw_sample_splitting=True,
                  apply_cross_fitting=True):
        obj = cls(obj_dml_data,
                  ml_g,
                  ml_m,
                  ml_r,
                  n_folds,
                  n_rep_cross_fit,
                  score,
                  dml_procedure,
                  draw_sample_splitting,
                  apply_cross_fitting)
        obj.partialX = True
        obj.partialZ = True
        obj._initialize_ml_nuisance_params()
        return obj

    @property
    def g_params(self):
        return self._g_params

    @property
    def m_params(self):
        return self._m_params

    @property
    def m_params_mult_instr(self):
        return self._m_params_mult_instr

    @property
    def r_params(self):
        return self._r_params

    # The private properties with __ always deliver the single treatment, single (cross-fitting) sample subselection
    # The slicing is based on the two properties self._i_treat, the index of the treatment variable, and
    # self._i_rep, the index of the cross-fitting sample.

    @property
    def __g_params(self):
        return self._g_params[self.d_cols[self._i_treat]][self._i_rep]

    @property
    def __m_params(self):
        return self._m_params[self.d_cols[self._i_treat]][self._i_rep]

    @property
    def __m_params_mult_instr(self):
        return self._m_params_mult_instr[self.z_cols[self._i_instr]][self.d_cols[self._i_treat]][self._i_rep]

    @property
    def __r_params(self):
        return self._r_params[self.d_cols[self._i_treat]][self._i_rep]

    def _check_score(self, score):
        if isinstance(score, str):
            valid_score = ['partialling out', 'partialling out']
            # check whether its worth implementing the IV_type as well
            # In CCDHNR equation (4.7) a score of this type is provided;
            # however in the following paragraph it is explained that one might
            # still need to estimate the partialling out type first
            if score not in valid_score:
                raise ValueError('invalid score ' + score +
                                 '\n valid score ' + valid_score)
        else:
            if not callable(score):
                raise ValueError('score should be either a string or a callable.'
                                 ' %r was passed' % score)
        return score

    def _check_data(self, obj_dml_data):
        return

    def _ml_nuisance_and_score_elements(self, obj_dml_data, smpls, n_jobs_cv):
        if self.partialX & (not self.partialZ):
            psi_a, psi_b = self._ml_nuisance_and_score_elements_partialX(obj_dml_data, smpls, n_jobs_cv)
        elif (not self.partialX) & self.partialZ:
            psi_a, psi_b = self._ml_nuisance_and_score_elements_partialZ(obj_dml_data, smpls, n_jobs_cv)
        elif self.partialX & self.partialZ:
            psi_a, psi_b = self._ml_nuisance_and_score_elements_partialXZ(obj_dml_data, smpls, n_jobs_cv)

        return psi_a, psi_b

    def _ml_nuisance_tuning(self, obj_dml_data, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv):
        if self.partialX & (not self.partialZ):
            res = self._ml_nuisance_tuning_partialX(obj_dml_data, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv)
        elif (not self.partialX) & self.partialZ:
            res = self._ml_nuisance_tuning_partialZ(obj_dml_data, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv)
        elif self.partialX & self.partialZ:
            res = self._ml_nuisance_tuning_partialXZ(obj_dml_data, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv)

        return res

    def _ml_nuisance_and_score_elements_partialX(self, obj_dml_data, smpls, n_jobs_cv):
        X, y = check_X_y(obj_dml_data.x, obj_dml_data.y)
        X, d = check_X_y(X, obj_dml_data.d)
        
        # nuisance g
        g_hat = _dml_cv_predict(self.ml_g, X, y, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self.__g_params)
        
        # nuisance m
        if obj_dml_data.n_instr == 1:
            # one instrument: just identified
            X, z = check_X_y(X, obj_dml_data.z)
            m_hat = _dml_cv_predict(self.ml_m, X, z, smpls=smpls, n_jobs=n_jobs_cv,
                                    est_params=self.__m_params)
        else:
            # several instruments: 2SLS
            m_hat = np.full((self.n_obs, obj_dml_data.n_instr), np.nan)
            z = obj_dml_data.z
            for i_instr in range(obj_dml_data.n_instr):
                self._i_instr = i_instr
                X, this_z = check_X_y(X, z[:, i_instr])
                m_hat[:, i_instr] = _dml_cv_predict(self.ml_m, X, this_z, smpls=smpls, n_jobs=n_jobs_cv,
                                                    est_params=self.__m_params_mult_instr)

        # nuisance r
        r_hat = _dml_cv_predict(self.ml_r, X, d, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self.__r_params)
        
        # compute residuals
        u_hat = y - g_hat
        w_hat = d - r_hat
        v_hat = z - m_hat

        if obj_dml_data.n_instr > 1:
            assert self.apply_cross_fitting
            # TODO check whether the no cross-fitting case can be supported here
            # projection of r_hat on m_hat
            r_hat_tilde = _dml_cv_predict(LinearRegression(fit_intercept=True), v_hat, w_hat,
                                          smpls=smpls, n_jobs=n_jobs_cv)
        score = self.score
        self._check_score(score)
        if isinstance(self.score, str):
            if obj_dml_data.n_instr == 1:
                psi_a = -np.multiply(w_hat, v_hat)
                psi_b = np.multiply(v_hat, u_hat)
            else:
                psi_a = -np.multiply(w_hat, r_hat_tilde)
                psi_b = np.multiply(r_hat_tilde, u_hat)
        elif callable(self.score):
            assert obj_dml_data.n_instr == 1, 'callable score not implemented for DoubleMLPLIV.partialX with several instruments'
            psi_a, psi_b = self.score(y, z, d,
                                      g_hat, m_hat, r_hat, smpls)

        return psi_a, psi_b

    def _ml_nuisance_and_score_elements_partialZ(self, obj_dml_data, smpls, n_jobs_cv):
        y = obj_dml_data.y
        XZ, d = check_X_y(np.hstack((obj_dml_data.x, obj_dml_data.z)),
                          obj_dml_data.d)

        # nuisance m
        r_hat = _dml_cv_predict(self.ml_r, XZ, d, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self.__r_params)

        score = self.score
        self._check_score(score)
        if isinstance(self.score, str):
            psi_a = -np.multiply(r_hat, d)
            psi_b = np.multiply(r_hat, y)
        elif callable(self.score):
            assert obj_dml_data.n_instr == 1, 'callable score not implemented for DoubleMLPLIV.partialZ'

        return psi_a, psi_b

    def _ml_nuisance_and_score_elements_partialXZ(self, obj_dml_data, smpls, n_jobs_cv):
        X, y = check_X_y(obj_dml_data.x, obj_dml_data.y)
        XZ, d = check_X_y(np.hstack((obj_dml_data.x, obj_dml_data.z)),
                          obj_dml_data.d)
        X, d = check_X_y(X, obj_dml_data.d)

        # nuisance g
        g_hat = _dml_cv_predict(self.ml_g, X, y, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self.__g_params)

        # nuisance m
        m_hat, m_hat_on_train = _dml_cv_predict(self.ml_m, XZ, d, smpls=smpls, n_jobs=n_jobs_cv,
                                                est_params=self.__m_params, return_train_preds=True)

        # nuisance r
        m_hat_tilde = _dml_cv_predict(self.ml_r, X, m_hat_on_train, smpls=smpls, n_jobs=n_jobs_cv,
                                      est_params=self.__r_params)

        # compute residuals
        u_hat = y - g_hat
        w_hat = d - m_hat_tilde

        score = self.score
        self._check_score(score)
        if isinstance(self.score, str):
            psi_a = -np.multiply(w_hat, (m_hat-m_hat_tilde))
            psi_b = np.multiply((m_hat-m_hat_tilde), u_hat)
        elif callable(self.score):
            assert obj_dml_data.n_instr == 1, 'callable score not implemented for DoubleMLPLIV.partialXZ'

        return psi_a, psi_b

    def _ml_nuisance_tuning_partialX(self, obj_dml_data, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv):
        X, y = check_X_y(obj_dml_data.x, obj_dml_data.y)
        X, d = check_X_y(X, obj_dml_data.d)

        if scoring_methods is None:
            scoring_methods = {'scoring_methods_g': None,
                               'scoring_methods_m': None,
                               'scoring_methods_r': None}

        g_tune_res = [None] * len(smpls)
        if obj_dml_data.n_instr > 1:
            m_tune_res = {instr_var: [None] * len(smpls) for instr_var in self.z_cols}
        else:
            m_tune_res = [None] * len(smpls)
        r_tune_res = [None] * len(smpls)

        for idx, (train_index, test_index) in enumerate(smpls):
            # cv for ml_g
            g_tune_resampling = KFold(n_splits=n_folds_tune)
            g_grid_search = GridSearchCV(self.ml_g, param_grids['param_grid_g'],
                                         scoring=scoring_methods['scoring_methods_g'],
                                         cv=g_tune_resampling)
            g_tune_res[idx] = g_grid_search.fit(X[train_index, :], y[train_index])

            # cv for ml_m
            if obj_dml_data.n_instr > 1:
                # several instruments: 2SLS
                z = obj_dml_data.z
                for i_instr in range(obj_dml_data.n_instr):
                    X, this_z = check_X_y(X, z[:, i_instr])
                    m_tune_resampling = KFold(n_splits=n_folds_tune)
                    m_grid_search = GridSearchCV(self.ml_m, param_grids['param_grid_m'],
                                                 scoring=scoring_methods['scoring_methods_m'],
                                                 cv=m_tune_resampling)
                    m_tune_res[self.z_cols[i_instr]][idx] = m_grid_search.fit(X[train_index, :], this_z[train_index])
            else:
                # one instrument: just identified
                X, z = check_X_y(X, obj_dml_data.z)
                m_tune_resampling = KFold(n_splits=n_folds_tune)
                m_grid_search = GridSearchCV(self.ml_m, param_grids['param_grid_m'],
                                             scoring=scoring_methods['scoring_methods_m'],
                                             cv=m_tune_resampling)
                m_tune_res[idx] = m_grid_search.fit(X[train_index, :], z[train_index])

            # cv for ml_r
            r_tune_resampling = KFold(n_splits=n_folds_tune)
            r_grid_search = GridSearchCV(self.ml_r, param_grids['param_grid_r'],
                                         scoring=scoring_methods['scoring_methods_r'],
                                         cv=r_tune_resampling)
            r_tune_res[idx] = r_grid_search.fit(X[train_index, :], d[train_index])

        g_best_params = [xx.best_params_ for xx in g_tune_res]
        r_best_params = [xx.best_params_ for xx in r_tune_res]
        if obj_dml_data.n_instr > 1:
            params = {'ml_g': g_best_params,
                      'ml_r': r_best_params}
            for instr_var in self.z_cols:
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

    def _ml_nuisance_tuning_partialZ(self, obj_dml_data, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv):
        XZ, d = check_X_y(np.hstack((obj_dml_data.x, obj_dml_data.z)),
                          obj_dml_data.d)

        if scoring_methods is None:
            scoring_methods = {'scoring_methods_r': None}

        m_tune_res = [None] * len(smpls)

        for idx, (train_index, test_index) in enumerate(smpls):
            # cv for ml_m
            m_tune_resampling = KFold(n_splits=n_folds_tune)
            m_grid_search = GridSearchCV(self.ml_r, param_grids['param_grid_r'],
                                         scoring=scoring_methods['scoring_methods_r'],
                                         cv=m_tune_resampling)
            m_tune_res[idx] = m_grid_search.fit(XZ[train_index, :], d[train_index])

        m_best_params = [xx.best_params_ for xx in m_tune_res]

        params = {'ml_r': m_best_params}

        tune_res = {'r_tune': m_tune_res}

        res = {'params': params,
               'tune_res': tune_res}

        return res

    def _ml_nuisance_tuning_partialXZ(self, obj_dml_data, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv):
        X, y = check_X_y(obj_dml_data.x, obj_dml_data.y)
        XZ, d = check_X_y(np.hstack((obj_dml_data.x, obj_dml_data.z)),
                          obj_dml_data.d)
        X, d = check_X_y(X, obj_dml_data.d)

        if scoring_methods is None:
            scoring_methods = {'scoring_methods_g': None,
                               'scoring_methods_m': None,
                               'scoring_methods_r': None}

        g_tune_res = [None] * len(smpls)
        m_tune_res = [None] * len(smpls)
        r_tune_res = [None] * len(smpls)

        for idx, (train_index, test_index) in enumerate(smpls):
            # cv for ml_g
            g_tune_resampling = KFold(n_splits=n_folds_tune)
            g_grid_search = GridSearchCV(self.ml_g, param_grids['param_grid_g'],
                                         scoring=scoring_methods['scoring_methods_g'],
                                         cv=g_tune_resampling)
            g_tune_res[idx] = g_grid_search.fit(X[train_index, :], y[train_index])

            # cv for ml_m
            m_tune_resampling = KFold(n_splits=n_folds_tune)
            m_grid_search = GridSearchCV(self.ml_m, param_grids['param_grid_m'],
                                         scoring=scoring_methods['scoring_methods_m'],
                                         cv=m_tune_resampling)
            m_tune_res[idx] = m_grid_search.fit(XZ[train_index, :], d[train_index])

            # cv for ml_r
            m_hat = m_grid_search.predict(XZ[train_index, :])
            r_tune_resampling = KFold(n_splits=n_folds_tune)
            r_grid_search = GridSearchCV(self.ml_r, param_grids['param_grid_r'],
                                         scoring=scoring_methods['scoring_methods_r'],
                                         cv=r_tune_resampling)
            r_tune_res[idx] = r_grid_search.fit(X[train_index, :], m_hat)

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

    def _initialize_ml_nuisance_params(self):
        if self.partialX & (not self.partialZ):
            self._g_params = {key: [None] * self.n_rep_cross_fit for key in self.d_cols}
            self._r_params = {key: [None] * self.n_rep_cross_fit for key in self.d_cols}
            if self.n_instr == 1:
                self._m_params = {key_d: [None] * self.n_rep_cross_fit for key_d in self.d_cols}
            else:
                self._m_params_mult_instr = {key_z:  {key_d: [None] * self.n_rep_cross_fit for key_d in self.d_cols} for key_z in self.z_cols}
        elif (not self.partialX) & self.partialZ:
            self._g_params = None
            self._m_params = None
            self._r_params = {key: [None] * self.n_rep_cross_fit for key in self.d_cols}
        elif self.partialX & self.partialZ:
            self._g_params = {key: [None] * self.n_rep_cross_fit for key in self.d_cols}
            self._m_params = {key: [None] * self.n_rep_cross_fit for key in self.d_cols}
            self._r_params = {key: [None] * self.n_rep_cross_fit for key in self.d_cols}

    def set_ml_nuisance_params(self, learner, treat_var, params):
        if self.partialX & (not self.partialZ):
            if self.n_instr == 1:
                valid_learner = ['ml_g', 'ml_m', 'ml_r']
            else:
                valid_learner = ['ml_g', 'ml_r'] + ['ml_m_' + z_col for z_col in self.z_cols]
        elif (not self.partialX) & self.partialZ:
            valid_learner = ['ml_r']
        elif self.partialX & self.partialZ:
            valid_learner = ['ml_g', 'ml_m', 'ml_r']

        if learner not in valid_learner:
            raise ValueError('invalid nuisance learner' + learner +
                             '\n valid nuisance learner ' + ' or '.join(valid_learner))
        if treat_var not in self.d_cols:
            raise ValueError('invalid treatment variable' + learner +
                             '\n valid treatment variable ' + ' or '.join(self.d_cols))

        if isinstance(params, dict):
            all_params = [[params] * self.n_folds] * self.n_rep_cross_fit
        else:
            assert len(params) == self.n_rep_cross_fit
            assert np.all(np.array([len(x) for x in params]) == self.n_folds)
            all_params = params

        if learner == 'ml_g':
            self._g_params[treat_var] = all_params
        elif learner == 'ml_r':
            self._r_params[treat_var] = all_params
        elif learner == 'ml_m':
            self._m_params[treat_var] = all_params
        else:
            assert learner.startswith('ml_m_')
            instr_var = learner.split('ml_m_')[1]
            self._m_params_mult_instr[instr_var][treat_var] = all_params
