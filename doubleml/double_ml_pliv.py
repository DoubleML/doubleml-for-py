import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

from .double_ml import DoubleML, DoubleMLData
from .helper import _dml_cross_val_predict


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
        self._g_params = None
        self._m_params = None
        self._r_params = None

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
        return obj

    def _check_score(self, score):
        if isinstance(score, str):
            valid_score = ['partialling out']
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
        g_hat = _dml_cross_val_predict(self.ml_g, X, y, smpls=smpls, n_jobs=n_jobs_cv)
        
        # nuisance m
        if obj_dml_data.n_instr == 1:
            # one instrument: just identified
            X, z = check_X_y(X, obj_dml_data.z)
            m_hat = _dml_cross_val_predict(self.ml_m, X, z, smpls=smpls, n_jobs=n_jobs_cv)
        else:
            # several instruments: 2SLS
            m_hat = np.full((self.n_obs_test, obj_dml_data.n_instr), np.nan)
            for i_instr in range(obj_dml_data.n_instr):
                X, this_z = check_X_y(X, obj_dml_data.z[:, i_instr])
                m_hat[:, i_instr] = _dml_cross_val_predict(self.ml_m, X, this_z, smpls=smpls, n_jobs=n_jobs_cv)

        # nuisance r
        r_hat = _dml_cross_val_predict(self.ml_r, X, d, smpls=smpls, n_jobs=n_jobs_cv)

        if self.apply_cross_fitting:
            y_test = y
            d_test = d
            z_test = obj_dml_data.z
        else:
            # the no cross-fitting case
            test_index = self.smpls[0][0][1]
            y_test = y[test_index]
            d_test = d[test_index]
            z_test = obj_dml_data.z[test_index]
        
        # compute residuals
        u_hat = y_test - g_hat
        w_hat = d_test - r_hat
        v_hat = z_test - m_hat

        if obj_dml_data.n_instr > 1:
            assert self.apply_cross_fitting
            # TODO check whether the no cross-fitting case can be supported here
            # projection of r_hat on m_hat
            r_hat_tilde = _dml_cross_val_predict(LinearRegression(fit_intercept=True), v_hat, w_hat,
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
            psi_a, psi_b = self.score(y_test, z_test, d_test,
                                      g_hat, m_hat, r_hat, smpls)

        return psi_a, psi_b

    def _ml_nuisance_and_score_elements_partialZ(self, obj_dml_data, smpls, n_jobs_cv):
        y = obj_dml_data.y
        XZ, d = check_X_y(np.hstack((obj_dml_data.x, obj_dml_data.z)),
                          obj_dml_data.d)

        # nuisance m
        r_hat = _dml_cross_val_predict(self.ml_r, XZ, d, smpls=smpls, n_jobs=n_jobs_cv)

        if self.apply_cross_fitting:
            y_test = y
            d_test = d
        else:
            # the no cross-fitting case
            test_index = self.smpls[0][0][1]
            y_test = y[test_index]
            d_test = d[test_index]

        score = self.score
        self._check_score(score)
        if isinstance(self.score, str):
            psi_a = -np.multiply(r_hat, d_test)
            psi_b = np.multiply(r_hat, y_test)
        elif callable(self.score):
            assert obj_dml_data.n_instr == 1, 'callable score not implemented for DoubleMLPLIV.partialZ'

        return psi_a, psi_b

    def _ml_nuisance_and_score_elements_partialXZ(self, obj_dml_data, smpls, n_jobs_cv):
        X, y = check_X_y(obj_dml_data.x, obj_dml_data.y)
        XZ, d = check_X_y(np.hstack((obj_dml_data.x, obj_dml_data.z)),
                          obj_dml_data.d)
        X, d = check_X_y(X, obj_dml_data.d)

        # nuisance g
        g_hat = _dml_cross_val_predict(self.ml_g, X, y, smpls=smpls, n_jobs=n_jobs_cv)

        # nuisance m
        m_hat = _dml_cross_val_predict(self.ml_m, XZ, d, smpls=smpls, n_jobs=n_jobs_cv)

        # nuisance r
        m_hat_tilde = _dml_cross_val_predict(self.ml_r, X, m_hat, smpls=smpls, n_jobs=n_jobs_cv)

        if self.apply_cross_fitting:
            y_test = y
            d_test = d
        else:
            # the no cross-fitting case
            test_index = self.smpls[0][0][1]
            y_test = y[test_index]
            d_test = d[test_index]

        # compute residuals
        u_hat = y_test - g_hat
        w_hat = d_test - m_hat_tilde

        score = self.score
        self._check_score(score)
        if isinstance(self.score, str):
            psi_a = -np.multiply(w_hat, (m_hat-m_hat_tilde))
            psi_b = np.multiply((m_hat-m_hat_tilde), u_hat)
        elif callable(self.score):
            assert obj_dml_data.n_instr == 1, 'callable score not implemented for DoubleMLPLIV.partialXZ'

        return psi_a, psi_b

    def _ml_nuisance_tuning_partialX(self, obj_dml_data, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv):
        assert obj_dml_data.n_instr == 1, 'tuning not implemented for several instruments'
        X, y = check_X_y(obj_dml_data.x, obj_dml_data.y)
        X, z = check_X_y(X, obj_dml_data.z)
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
            m_tune_res[idx] = m_grid_search.fit(X[train_index, :], z[train_index])

            # cv for ml_r
            r_tune_resampling = KFold(n_splits=n_folds_tune)
            r_grid_search = GridSearchCV(self.ml_r, param_grids['param_grid_r'],
                                         scoring=scoring_methods['scoring_methods_r'],
                                         cv=r_tune_resampling)
            r_tune_res[idx] = r_grid_search.fit(X[train_index, :], d[train_index])

        g_best_params = [xx.best_params_ for xx in g_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]
        r_best_params = [xx.best_params_ for xx in r_tune_res]

        params = {'g_params': g_best_params,
                  'm_params': m_best_params,
                  'r_params': r_best_params}

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
            scoring_methods = {'scoring_methods_m': None}

        m_tune_res = [None] * len(smpls)

        for idx, (train_index, test_index) in enumerate(smpls):
            # cv for ml_m
            m_tune_resampling = KFold(n_splits=n_folds_tune)
            m_grid_search = GridSearchCV(self.ml_m, param_grids['param_grid_m'],
                                         scoring=scoring_methods['scoring_methods_m'],
                                         cv=m_tune_resampling)
            m_tune_res[idx] = m_grid_search.fit(XZ[train_index, :], d[train_index])

        m_best_params = [xx.best_params_ for xx in m_tune_res]

        params = {'m_params': m_best_params}

        tune_res = {'m_tune': m_tune_res}

        res = {'params': params,
               'tune_res': tune_res}

        return res

    def _ml_nuisance_tuning_partialXZ(self, obj_dml_data, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv):
        return

    def _set_ml_nuisance_params(self, params):
        self._g_params = params['g_params']
        self._m_params = params['m_params']
        self._r_params = params['r_params']

