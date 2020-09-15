import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from .double_ml import DoubleML
from .helper import _dml_cross_val_predict


class DoubleMLPLR(DoubleML):
    """
    Double machine learning for partially linear regression models

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
    inf_model :
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
    >>> from doubleml.datasets import make_plr_data
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.base import clone
    >>> learner = RandomForestRegressor(max_depth=2, n_estimators=10)
    >>> ml_learners = {'ml_m': clone(learner), 'ml_g': clone(learner)}
    >>> data = make_plr_data()
    >>> obj_dml_data = dml.DoubleMLData(data, 'y', 'd')
    >>> dml_plr_obj = dml.DoubleMLPLR(obj_dml_data, ml_learners)
    >>> dml_plr_obj.fit()
    >>> dml_plr_obj.summary
         coef   std err         t         P>|t|    2.5 %    97.5 %
    d  0.608353  0.101595  5.988004  2.124312e-09  0.40923  0.807477

    Notes
    -----
    .. include:: ../../shared/models/plr.rst
    """
    def __init__(self,
                 obj_dml_data,
                 ml_learners,
                 n_folds=5,
                 n_rep_cross_fit=1,
                 inf_model='DML2018',
                 dml_procedure='dml1',
                 draw_sample_splitting=True,
                 apply_cross_fitting=True):
        super().__init__(obj_dml_data,
                         ml_learners,
                         n_folds,
                         n_rep_cross_fit,
                         inf_model,
                         dml_procedure,
                         draw_sample_splitting,
                         apply_cross_fitting)
        self._g_params = None
        self._m_params = None

    def _check_inf_method(self, inf_model):
        if isinstance(inf_model, str):
            valid_inf_model = ['IV-type', 'DML2018']
            if inf_model not in valid_inf_model:
                raise ValueError('invalid inf_model ' + inf_model +
                                 '\n valid inf_model ' + ' or '.join(valid_inf_model))
        else:
            if not callable(inf_model):
                raise ValueError('inf_model should be either a string or a callable.'
                                 ' %r was passed' % inf_model)
        return inf_model

    def _check_data(self, obj_dml_data):
        assert obj_dml_data.z_col is None
        return

    def _ml_nuisance_and_score_elements(self, obj_dml_data, smpls, n_jobs_cv):
        ml_g = self.ml_learners['ml_g']
        ml_m = self.ml_learners['ml_m']
        
        X, y = check_X_y(obj_dml_data.x, obj_dml_data.y)
        X, d = check_X_y(X, obj_dml_data.d)
        
        # nuisance g
        g_hat = _dml_cross_val_predict(ml_g, X, y, smpls=smpls, n_jobs=n_jobs_cv,
                                       est_params=self._g_params)
        
        # nuisance m
        m_hat = _dml_cross_val_predict(ml_m, X, d, smpls=smpls, n_jobs=n_jobs_cv,
                                       est_params=self._m_params)

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
        v_hat = d_test - m_hat
        v_hatd = np.multiply(v_hat, d_test)

        inf_model = self.inf_model
        self._check_inf_method(inf_model)
        if isinstance(self.inf_model, str):
            if inf_model == 'IV-type':
                score_a = -v_hatd
            elif inf_model == 'DML2018':
                score_a = -np.multiply(v_hat, v_hat)
            score_b = np.multiply(v_hat, u_hat)
        elif callable(self.inf_model):
            score_a, score_b = self.inf_model(y_test, d_test, g_hat, m_hat, smpls)
        
        return score_a, score_b

    def _ml_nuisance_tuning(self, obj_dml_data, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv):
        ml_g = self.ml_learners['ml_g']
        ml_m = self.ml_learners['ml_m']

        X, y = check_X_y(obj_dml_data.x, obj_dml_data.y)
        X, d = check_X_y(X, obj_dml_data.d)

        if scoring_methods is None:
            scoring_methods = {'scoring_methods_g': None,
                               'scoring_methods_m': None}

        g_tune_res = [None] * len(smpls)
        m_tune_res = [None] * len(smpls)

        for idx, (train_index, test_index) in enumerate(smpls):
            # cv for ml_g
            g_tune_resampling = KFold(n_splits=n_folds_tune)
            g_grid_search = GridSearchCV(ml_g, param_grids['param_grid_g'],
                                         scoring=scoring_methods['scoring_methods_g'],
                                         cv=g_tune_resampling)
            g_tune_res[idx] = g_grid_search.fit(X[train_index, :], y[train_index])

            # cv for ml_m
            m_tune_resampling = KFold(n_splits=n_folds_tune)
            m_grid_search = GridSearchCV(ml_m, param_grids['param_grid_m'],
                                         scoring=scoring_methods['scoring_methods_m'],
                                         cv=m_tune_resampling)
            m_tune_res[idx] = m_grid_search.fit(X[train_index, :], d[train_index])

        g_best_params = [xx.best_params_ for xx in g_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]

        params = {'g_params': g_best_params,
                  'm_params': m_best_params}

        tune_res = {'g_tune': g_tune_res,
                    'm_tune': m_tune_res}

        res = {'params': params,
               'tune_res': tune_res}

        return(res)

    def _set_ml_nuisance_params(self, params):
        self._g_params = params['g_params']
        self._m_params = params['m_params']
