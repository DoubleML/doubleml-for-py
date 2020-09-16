import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

from .double_ml import DoubleML, DoubleMLData
from .helper import _dml_cross_val_predict


class DoubleMLPLIVselectZ(DoubleML):
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
    >>>

    Notes
    -----
    """
    def __init__(self,
                 obj_dml_data,
                 ml_m,
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
        self.ml_m = ml_m
        self._m_params = None

    def _check_score(self, score):
        if isinstance(score, str):
            valid_score = ['partialling out']
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
        y = obj_dml_data.y
        XZ, d = check_X_y(np.hstack((obj_dml_data.x, obj_dml_data.z)),
                          obj_dml_data.d)

        # nuisance m
        m_hat = _dml_cross_val_predict(self.ml_m, XZ, d, smpls=smpls, n_jobs=n_jobs_cv)

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
            psi_a = -np.multiply(m_hat, d_test)
            psi_b = np.multiply(m_hat, y_test)
        elif callable(self.score):
            assert obj_dml_data.n_instr == 1, 'callable score not implemented for several instruments'
            psi_a, psi_b = self.score(y_test, d_test,
                                      m_hat, smpls)

        return psi_a, psi_b

    def _ml_nuisance_tuning(self, obj_dml_data, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv):
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

        return(res)

    def _set_ml_nuisance_params(self, params):
        self._m_params = params['m_params']

