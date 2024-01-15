import numpy as np
import pytest
import math

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

import doubleml as dml
from doubleml.datasets import make_pliv_multiway_cluster_CKMS2021

from ._utils import _clone
from ._utils_cluster import var_one_way_cluster, est_one_way_cluster_dml2, \
    est_two_way_cluster_dml2, var_two_way_cluster
from ..plm.tests._utils_pliv_manual import fit_pliv, compute_pliv_residuals

np.random.seed(1234)
# Set the simulation parameters
N = 25  # number of observations (first dimension)
M = 25  # number of observations (second dimension)
dim_x = 100  # dimension of x

obj_dml_cluster_data = make_pliv_multiway_cluster_CKMS2021(N, M, dim_x)

obj_dml_oneway_cluster_data = make_pliv_multiway_cluster_CKMS2021(N, M, dim_x,
                                                                  omega_X=np.array([0.25, 0]),
                                                                  omega_epsilon=np.array([0.25, 0]),
                                                                  omega_v=np.array([0.25, 0]),
                                                                  omega_V=np.array([0.25, 0]))
# only the first cluster variable is relevant with the weight setting above
obj_dml_oneway_cluster_data.cluster_cols = 'cluster_var_i'


@pytest.fixture(scope='module',
                params=[RandomForestRegressor(max_depth=2, n_estimators=10),
                        LinearRegression(),
                        Lasso(alpha=0.1)])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['partialling out', 'IV-type'])
def score(request):
    return request.param


@pytest.fixture(scope='module')
def dml_pliv_multiway_cluster_fixture(generate_data_iv, learner, score):
    n_folds = 2
    n_rep = 2

    # Set machine learning methods for l, m, r & g
    ml_l = _clone(learner)
    ml_m = _clone(learner)
    ml_r = _clone(learner)
    if score == 'IV-type':
        ml_g = _clone(learner)
    else:
        ml_g = None

    np.random.seed(3141)
    dml_pliv_obj = dml.DoubleMLPLIV(obj_dml_cluster_data,
                                    ml_l, ml_m, ml_r, ml_g,
                                    n_folds=n_folds,
                                    n_rep=n_rep,
                                    score=score)

    np.random.seed(3141)
    dml_pliv_obj.fit()

    np.random.seed(3141)
    y = obj_dml_cluster_data.y
    x = obj_dml_cluster_data.x
    d = obj_dml_cluster_data.d
    z = np.ravel(obj_dml_cluster_data.z)

    res_manual = fit_pliv(y, x, d, z,
                          _clone(learner), _clone(learner), _clone(learner), _clone(learner),
                          dml_pliv_obj.smpls, score,
                          n_rep=n_rep)
    thetas = np.full(n_rep, np.nan)
    ses = np.full(n_rep, np.nan)
    for i_rep in range(n_rep):
        l_hat = res_manual['all_l_hat'][i_rep]
        m_hat = res_manual['all_m_hat'][i_rep]
        r_hat = res_manual['all_r_hat'][i_rep]
        g_hat = res_manual['all_g_hat'][i_rep]
        smpls_one_split = dml_pliv_obj.smpls[i_rep]
        y_minus_l_hat, z_minus_m_hat, d_minus_r_hat, y_minus_g_hat = compute_pliv_residuals(
            y, d, z, l_hat, m_hat, r_hat, g_hat, smpls_one_split)

        if score == 'partialling out':
            psi_a = -np.multiply(z_minus_m_hat, d_minus_r_hat)
            psi_b = np.multiply(z_minus_m_hat, y_minus_l_hat)
            theta = est_two_way_cluster_dml2(psi_a, psi_b,
                                             obj_dml_cluster_data.cluster_vars[:, 0],
                                             obj_dml_cluster_data.cluster_vars[:, 1],
                                             smpls_one_split)

            psi = np.multiply(y_minus_l_hat - d_minus_r_hat * theta, z_minus_m_hat)
        else:
            assert score == 'IV-type'
            psi_a = -np.multiply(z_minus_m_hat, d)
            psi_b = np.multiply(z_minus_m_hat, y_minus_g_hat)
            theta = est_two_way_cluster_dml2(psi_a, psi_b,
                                             obj_dml_cluster_data.cluster_vars[:, 0],
                                             obj_dml_cluster_data.cluster_vars[:, 1],
                                             smpls_one_split)

            psi = np.multiply(y_minus_g_hat - d * theta, z_minus_m_hat)

        var = var_two_way_cluster(psi, psi_a,
                                  obj_dml_cluster_data.cluster_vars[:, 0],
                                  obj_dml_cluster_data.cluster_vars[:, 1],
                                  smpls_one_split)
        se = np.sqrt(var)
        thetas[i_rep] = theta
        ses[i_rep] = se[0]

    theta = np.median(thetas)
    n_clusters1 = len(np.unique(obj_dml_cluster_data.cluster_vars[:, 0]))
    n_clusters2 = len(np.unique(obj_dml_cluster_data.cluster_vars[:, 1]))
    var_scaling_factor = min(n_clusters1, n_clusters2)
    se = np.sqrt(np.median(np.power(ses, 2) * var_scaling_factor + np.power(thetas - theta, 2)) / var_scaling_factor)

    res_dict = {'coef': dml_pliv_obj.coef,
                'se': dml_pliv_obj.se,
                'coef_manual': theta,
                'se_manual': se}

    return res_dict


@pytest.mark.ci
def test_dml_pliv_multiway_cluster_coef(dml_pliv_multiway_cluster_fixture):
    assert math.isclose(dml_pliv_multiway_cluster_fixture['coef'][0],
                        dml_pliv_multiway_cluster_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_pliv_multiway_cluster_se(dml_pliv_multiway_cluster_fixture):
    assert math.isclose(dml_pliv_multiway_cluster_fixture['se'][0],
                        dml_pliv_multiway_cluster_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.fixture(scope='module')
def dml_pliv_oneway_cluster_fixture(generate_data_iv, learner, score):
    n_folds = 3

    # Set machine learning methods for l, m, r & g
    ml_l = _clone(learner)
    ml_m = _clone(learner)
    ml_r = _clone(learner)
    if score == 'IV-type':
        ml_g = _clone(learner)
    else:
        ml_g = None

    np.random.seed(3141)
    dml_pliv_obj = dml.DoubleMLPLIV(obj_dml_oneway_cluster_data,
                                    ml_l, ml_m, ml_r, ml_g,
                                    n_folds=n_folds,
                                    score=score)

    np.random.seed(3141)
    dml_pliv_obj.fit()

    np.random.seed(3141)
    y = obj_dml_oneway_cluster_data.y
    x = obj_dml_oneway_cluster_data.x
    d = obj_dml_oneway_cluster_data.d
    z = np.ravel(obj_dml_oneway_cluster_data.z)

    res_manual = fit_pliv(y, x, d, z,
                          _clone(learner), _clone(learner), _clone(learner), _clone(learner),
                          dml_pliv_obj.smpls, score)
    l_hat = res_manual['all_l_hat'][0]
    m_hat = res_manual['all_m_hat'][0]
    r_hat = res_manual['all_r_hat'][0]
    g_hat = res_manual['all_g_hat'][0]
    smpls_one_split = dml_pliv_obj.smpls[0]
    y_minus_l_hat, z_minus_m_hat, d_minus_r_hat, y_minus_g_hat = compute_pliv_residuals(
        y, d, z, l_hat, m_hat, r_hat, g_hat, smpls_one_split)

    if score == 'partialling out':
        psi_a = -np.multiply(z_minus_m_hat, d_minus_r_hat)
        psi_b = np.multiply(z_minus_m_hat, y_minus_l_hat)
        theta = est_one_way_cluster_dml2(psi_a, psi_b,
                                         obj_dml_oneway_cluster_data.cluster_vars[:, 0],
                                         smpls_one_split)

        psi = np.multiply(y_minus_l_hat - d_minus_r_hat * theta, z_minus_m_hat)
    else:
        assert score == 'IV-type'
        psi_a = -np.multiply(z_minus_m_hat, d)
        psi_b = np.multiply(z_minus_m_hat, y_minus_g_hat)
        theta = est_one_way_cluster_dml2(psi_a, psi_b,
                                         obj_dml_oneway_cluster_data.cluster_vars[:, 0],
                                         smpls_one_split)

        psi = np.multiply(y_minus_g_hat - d * theta, z_minus_m_hat)

    var = var_one_way_cluster(psi, psi_a,
                              obj_dml_oneway_cluster_data.cluster_vars[:, 0],
                              smpls_one_split)
    se = np.sqrt(var)

    res_dict = {'coef': dml_pliv_obj.coef,
                'se': dml_pliv_obj.se,
                'coef_manual': theta,
                'se_manual': se}

    return res_dict


@pytest.mark.ci
def test_dml_pliv_oneway_cluster_coef(dml_pliv_oneway_cluster_fixture):
    assert math.isclose(dml_pliv_oneway_cluster_fixture['coef'][0],
                        dml_pliv_oneway_cluster_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_pliv_oneway_cluster_se(dml_pliv_oneway_cluster_fixture):
    assert math.isclose(dml_pliv_oneway_cluster_fixture['se'][0],
                        dml_pliv_oneway_cluster_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.fixture(scope="module")
def dml_plr_cluster_with_index(generate_data1, learner):
    # in the one-way cluster case with exactly one observation per cluster, we get the same result w & w/o clustering
    n_folds = 2

    # collect data
    data = generate_data1
    x_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for m & l
    ml_l = _clone(learner)
    ml_m = _clone(learner)

    obj_dml_data = dml.DoubleMLData(data, 'y', ['d'], x_cols)
    np.random.seed(3141)
    dml_plr_obj = dml.DoubleMLPLR(obj_dml_data,
                                  ml_l, ml_m,
                                  n_folds=n_folds)
    dml_plr_obj.fit()

    df = data.reset_index()
    dml_cluster_data = dml.DoubleMLClusterData(df,
                                               y_col='y',
                                               d_cols='d',
                                               x_cols=x_cols,
                                               cluster_cols='index')
    np.random.seed(3141)
    dml_plr_cluster_obj = dml.DoubleMLPLR(dml_cluster_data,
                                          ml_l, ml_m,
                                          n_folds=n_folds)
    dml_plr_cluster_obj.fit()

    res_dict = {'coef': dml_plr_obj.coef,
                'coef_manual': dml_plr_cluster_obj.coef,
                'se': dml_plr_obj.se,
                'se_manual': dml_plr_cluster_obj.se}

    return res_dict


@pytest.mark.ci
def test_dml_plr_cluster_with_index_coef(dml_plr_cluster_with_index):
    assert math.isclose(dml_plr_cluster_with_index['coef'][0],
                        dml_plr_cluster_with_index['coef_manual'][0],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_plr_cluster_with_index_se(dml_plr_cluster_with_index):
    assert math.isclose(dml_plr_cluster_with_index['se'][0],
                        dml_plr_cluster_with_index['se_manual'][0],
                        rel_tol=1e-9, abs_tol=1e-4)
