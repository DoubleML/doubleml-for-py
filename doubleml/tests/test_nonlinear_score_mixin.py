import numpy as np
import pytest
import math

from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target

import doubleml as dml
from doubleml.double_ml import DoubleML
from doubleml.datasets import make_pliv_multiway_cluster_CKMS2021
from doubleml._utils import _dml_cv_predict, _check_finite_predictions
from doubleml._double_ml_score_mixins import NonLinearScoreMixin


class DoubleMLPLRWithNonLinearScoreMixin(NonLinearScoreMixin, DoubleML):
    _coef_bounds = (-np.inf, np.inf)
    _coef_start_val = 3.0

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

        self._check_data(self._dml_data)
        self._check_score(self.score)
        _ = self._check_learner(ml_g, 'ml_g', regressor=True, classifier=False)
        _ = self._check_learner(ml_m, 'ml_m', regressor=True, classifier=False)
        self._learner = {'ml_g': ml_g, 'ml_m': ml_m}
        self._predict_method = {'ml_g': 'predict', 'ml_m': 'predict'}

        self._initialize_ml_nuisance_params()

    @property
    def _score_element_names(self):
        return ['psi_a', 'psi_b']

    def _compute_score(self, psi_elements, coef, inds=None):
        psi_a = psi_elements['psi_a']
        psi_b = psi_elements['psi_b']
        if inds is not None:
            psi_a = psi_a[inds]
            psi_b = psi_b[inds]
        psi = psi_a * coef + psi_b
        return psi

    def _compute_score_deriv(self, psi_elements, coef, inds=None):
        psi_a = psi_elements['psi_a']
        if inds is not None:
            psi_a = psi_a[inds]
        return psi_a

    def _initialize_ml_nuisance_params(self):
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in ['ml_g', 'ml_m']}

    def _check_score(self, score):
        pass

    def _check_data(self, obj_dml_data):
        pass

    def _ml_nuisance_and_score_elements(self, smpls, n_jobs_cv):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)

        # nuisance g
        g_hat = _dml_cv_predict(self._learner['ml_g'], x, y, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_g'), method=self._predict_method['ml_g'])
        _check_finite_predictions(g_hat, self._learner['ml_g'], 'ml_g', smpls)

        # nuisance m
        m_hat = _dml_cv_predict(self._learner['ml_m'], x, d, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_m'), method=self._predict_method['ml_m'])
        _check_finite_predictions(m_hat, self._learner['ml_m'], 'ml_m', smpls)

        psi_a, psi_b = self._score_elements(y, d, g_hat, m_hat, smpls)
        psi_elements = {'psi_a': psi_a,
                        'psi_b': psi_b}
        preds = {'ml_g': g_hat,
                 'ml_m': m_hat}

        return psi_elements, preds

    def _score_elements(self, y, d, g_hat, m_hat, smpls):
        # compute residuals
        u_hat = y - g_hat
        v_hat = d - m_hat
        v_hatd = np.multiply(v_hat, d)

        assert isinstance(self.score, str)
        if self.score == 'IV-type':
            psi_a = -v_hatd
        else:
            assert self.score == 'partialling out'
            psi_a = -np.multiply(v_hat, v_hat)
        psi_b = np.multiply(v_hat, u_hat)

        return psi_a, psi_b

    def _ml_nuisance_tuning(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                            search_mode, n_iter_randomized_search):
        pass


@pytest.fixture(scope='module',
                params=[LinearRegression()])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['IV-type', 'partialling out'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope='module',
                params=[(-np.inf, np.inf),
                        (0, 5)])
def coef_bounds(request):
    return request.param


@pytest.fixture(scope="module")
def dml_plr_w_nonlinear_mixin_fixture(generate_data1, learner, score, dml_procedure):
    n_folds = 3

    # collect data
    data = generate_data1
    x_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for m & g
    ml_g = clone(learner)
    ml_m = clone(learner)

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData(data, 'y', ['d'], x_cols)
    dml_plr_obj = dml.DoubleMLPLR(obj_dml_data,
                                  ml_g, ml_m,
                                  n_folds=n_folds,
                                  score=score,
                                  dml_procedure=dml_procedure)

    dml_plr_obj.fit()

    np.random.seed(3141)
    dml_plr_obj2 = DoubleMLPLRWithNonLinearScoreMixin(obj_dml_data,
                                                      ml_g, ml_m,
                                                      n_folds=n_folds,
                                                      score=score,
                                                      dml_procedure=dml_procedure)
    dml_plr_obj2._coef_bounds # use different settings to also unit test the solver for bounded problems
    dml_plr_obj2.fit()

    res_dict = {'coef': dml_plr_obj.coef,
                'coef2': dml_plr_obj2.coef,
                'se': dml_plr_obj.se,
                'se2': dml_plr_obj2.se}

    return res_dict


@pytest.mark.ci
def test_dml_plr_coef(dml_plr_w_nonlinear_mixin_fixture):
    assert math.isclose(dml_plr_w_nonlinear_mixin_fixture['coef'],
                        dml_plr_w_nonlinear_mixin_fixture['coef2'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_plr_se(dml_plr_w_nonlinear_mixin_fixture):
    assert math.isclose(dml_plr_w_nonlinear_mixin_fixture['se'],
                        dml_plr_w_nonlinear_mixin_fixture['se2'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_doubleml_cluster_not_implemented_exception():
    np.random.seed(3141)
    dml_data = make_pliv_multiway_cluster_CKMS2021()
    dml_data.z_cols = None
    ml_g = LinearRegression()
    ml_m = LinearRegression()
    dml_plr = DoubleMLPLRWithNonLinearScoreMixin(dml_data, ml_g, ml_m)
    msg = 'Estimation with clustering not implemented.'
    with pytest.raises(NotImplementedError, match=msg):
        _ = dml_plr.fit()
