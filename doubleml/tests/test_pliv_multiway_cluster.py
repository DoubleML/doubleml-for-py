import numpy as np
import pytest
import math

from sklearn.base import clone

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

import doubleml as dml
from doubleml.datasets import make_pliv_multiway_cluster_CKMS2019
from doubleml.double_ml_resampling import DoubleMLMultiwayResampling

from ._utils_pliv_manual import pliv_dml1, pliv_dml2, fit_nuisance_pliv

np.random.seed(1234)
# Set the simulation parameters
N = 25  # number of observations (first dimension)
M = 25  # number of observations (second dimension)
dim_x = 100  # dimension of x

obj_dml_data = make_pliv_multiway_cluster_CKMS2019(N, M, dim_x)


@pytest.fixture(scope='module',
                params=[RandomForestRegressor(max_depth=2, n_estimators=10),
                        LinearRegression(),
                        Lasso(alpha=0.1)])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['partialling out'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope='module')
def dml_pliv_multiway_cluster_fixture(generate_data_iv, learner, score, dml_procedure):
    n_folds = 3

    np.random.seed(1234)
    smpl_sizes = [N, M]
    obj_dml_multiway_resampling = DoubleMLMultiwayResampling(n_folds, smpl_sizes)
    _, smpls_lin_ind = obj_dml_multiway_resampling.split_samples()

    # Set machine learning methods for g, m & r
    ml_g = clone(learner)
    ml_m = clone(learner)
    ml_r = clone(learner)

    np.random.seed(3141)
    dml_pliv_obj = dml.DoubleMLPLIV(obj_dml_data,
                                    ml_g, ml_m, ml_r,
                                    n_folds,
                                    dml_procedure=dml_procedure,
                                    draw_sample_splitting=False)
    dml_pliv_obj.set_sample_splitting(smpls_lin_ind)

    dml_pliv_obj.fit()

    np.random.seed(3141)
    y = obj_dml_data.y
    x = obj_dml_data.x
    d = obj_dml_data.d
    z = np.ravel(obj_dml_data.z)

    g_hat, m_hat, r_hat = fit_nuisance_pliv(y, x, d, z,
                                            clone(learner), clone(learner), clone(learner),
                                            smpls_lin_ind)

    if dml_procedure == 'dml1':
        res_manual, _ = pliv_dml1(y, x, d,
                                  z,
                                  g_hat, m_hat, r_hat,
                                  smpls_lin_ind, score)
    else:
        assert dml_procedure == 'dml2'
        res_manual, _ = pliv_dml2(y, x, d,
                                  z,
                                  g_hat, m_hat, r_hat,
                                  smpls_lin_ind, score)

    res_dict = {'coef': dml_pliv_obj.coef,
                'coef_manual': res_manual}

    return res_dict


@pytest.mark.ci
def test_dml_pliv_multiway_cluster_coef(dml_pliv_multiway_cluster_fixture):
    assert math.isclose(dml_pliv_multiway_cluster_fixture['coef'],
                        dml_pliv_multiway_cluster_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)
