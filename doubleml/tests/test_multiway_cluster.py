import numpy as np
import pytest
import math

from sklearn.base import clone

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

import doubleml as dml
from doubleml.datasets import make_pliv_multiway_cluster_CKMS2021

from ._utils_cluster import DoubleMLMultiwayResampling, var_one_way_cluster, est_one_way_cluster_dml2,\
    est_two_way_cluster_dml2, var_two_way_cluster
from ._utils_pliv_manual import fit_pliv, compute_pliv_residuals

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
                params=['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope='module')
def dml_pliv_multiway_cluster_old_vs_new_fixture(generate_data_iv, learner):
    n_folds = 3
    dml_procedure = 'dml1'  # same results are only obtained for dml1

    np.random.seed(3141)
    smpl_sizes = [N, M]
    obj_dml_multiway_resampling = DoubleMLMultiwayResampling(n_folds, smpl_sizes)
    _, smpls_lin_ind = obj_dml_multiway_resampling.split_samples()

    # Set machine learning methods for g, m & r
    ml_g = clone(learner)
    ml_m = clone(learner)
    ml_r = clone(learner)

    df = obj_dml_cluster_data.data.set_index(['cluster_var_i', 'cluster_var_j'])
    obj_dml_data = dml.DoubleMLData(df,
                                    y_col=obj_dml_cluster_data.y_col,
                                    d_cols=obj_dml_cluster_data.d_cols,
                                    x_cols=obj_dml_cluster_data.x_cols,
                                    z_cols=obj_dml_cluster_data.z_cols)

    dml_pliv_obj = dml.DoubleMLPLIV(obj_dml_data,
                                    ml_g, ml_m, ml_r,
                                    n_folds,
                                    dml_procedure=dml_procedure,
                                    draw_sample_splitting=False)
    dml_pliv_obj.set_sample_splitting(smpls_lin_ind)

    dml_pliv_obj.fit()

    np.random.seed(3141)
    dml_pliv_obj_cluster = dml.DoubleMLPLIV(obj_dml_cluster_data,
                                            ml_g, ml_m, ml_r,
                                            n_folds,
                                            dml_procedure=dml_procedure)
    dml_pliv_obj_cluster.fit()

    res_dict = {'coef': dml_pliv_obj.coef,
                'coef_manual': dml_pliv_obj_cluster.coef}

    return res_dict


@pytest.mark.ci
def test_dml_pliv_multiway_cluster_old_vs_new_coef(dml_pliv_multiway_cluster_old_vs_new_fixture):
    assert math.isclose(dml_pliv_multiway_cluster_old_vs_new_fixture['coef'],
                        dml_pliv_multiway_cluster_old_vs_new_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.fixture(scope='module')
def dml_pliv_multiway_cluster_fixture(generate_data_iv, learner, dml_procedure):
    n_folds = 2
    n_rep = 2
    score = 'partialling out'

    # Set machine learning methods for g, m & r
    ml_g = clone(learner)
    ml_m = clone(learner)
    ml_r = clone(learner)

    np.random.seed(3141)
    dml_pliv_obj = dml.DoubleMLPLIV(obj_dml_cluster_data,
                                    ml_g, ml_m, ml_r,
                                    n_folds,
                                    n_rep=n_rep,
                                    score=score,
                                    dml_procedure=dml_procedure)

    np.random.seed(3141)
    dml_pliv_obj.fit()

    np.random.seed(3141)
    y = obj_dml_cluster_data.y
    x = obj_dml_cluster_data.x
    d = obj_dml_cluster_data.d
    z = np.ravel(obj_dml_cluster_data.z)

    res_manual = fit_pliv(y, x, d, z,
                          clone(learner), clone(learner), clone(learner),
                          dml_pliv_obj.smpls, dml_procedure, score,
                          n_rep=n_rep)
    thetas = np.full(n_rep, np.nan)
    ses = np.full(n_rep, np.nan)
    for i_rep in range(n_rep):
        g_hat = res_manual['all_g_hat'][i_rep]
        m_hat = res_manual['all_m_hat'][i_rep]
        r_hat = res_manual['all_r_hat'][i_rep]
        smpls_one_split = dml_pliv_obj.smpls[i_rep]
        u_hat, v_hat, w_hat = compute_pliv_residuals(y, d, z, g_hat, m_hat, r_hat, smpls_one_split)

        psi_a = -np.multiply(v_hat, w_hat)
        if dml_procedure == 'dml2':
            psi_b = np.multiply(v_hat, u_hat)
            theta = est_two_way_cluster_dml2(psi_a, psi_b,
                                             obj_dml_cluster_data.cluster_vars[:, 0],
                                             obj_dml_cluster_data.cluster_vars[:, 1],
                                             smpls_one_split)
        else:
            theta = res_manual['thetas'][i_rep]
        psi = np.multiply(u_hat - w_hat * theta, v_hat)
        var = var_two_way_cluster(psi, psi_a,
                                  obj_dml_cluster_data.cluster_vars[:, 0],
                                  obj_dml_cluster_data.cluster_vars[:, 1],
                                  smpls_one_split)
        se = np.sqrt(var)
        thetas[i_rep] = theta
        ses[i_rep] = se

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
    assert math.isclose(dml_pliv_multiway_cluster_fixture['coef'],
                        dml_pliv_multiway_cluster_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_pliv_multiway_cluster_se(dml_pliv_multiway_cluster_fixture):
    assert math.isclose(dml_pliv_multiway_cluster_fixture['se'],
                        dml_pliv_multiway_cluster_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.fixture(scope='module')
def dml_pliv_oneway_cluster_fixture(generate_data_iv, learner, dml_procedure):
    n_folds = 3
    score = 'partialling out'

    # Set machine learning methods for g, m & r
    ml_g = clone(learner)
    ml_m = clone(learner)
    ml_r = clone(learner)

    np.random.seed(3141)
    dml_pliv_obj = dml.DoubleMLPLIV(obj_dml_oneway_cluster_data,
                                    ml_g, ml_m, ml_r,
                                    n_folds,
                                    score=score,
                                    dml_procedure=dml_procedure)

    np.random.seed(3141)
    dml_pliv_obj.fit()

    np.random.seed(3141)
    y = obj_dml_oneway_cluster_data.y
    x = obj_dml_oneway_cluster_data.x
    d = obj_dml_oneway_cluster_data.d
    z = np.ravel(obj_dml_oneway_cluster_data.z)

    res_manual = fit_pliv(y, x, d, z,
                          clone(learner), clone(learner), clone(learner),
                          dml_pliv_obj.smpls, dml_procedure, score)
    g_hat = res_manual['all_g_hat'][0]
    m_hat = res_manual['all_m_hat'][0]
    r_hat = res_manual['all_r_hat'][0]
    smpls_one_split = dml_pliv_obj.smpls[0]
    u_hat, v_hat, w_hat = compute_pliv_residuals(y, d, z, g_hat, m_hat, r_hat, smpls_one_split)

    psi_a = -np.multiply(v_hat, w_hat)
    if dml_procedure == 'dml2':
        psi_b = np.multiply(v_hat, u_hat)
        theta = est_one_way_cluster_dml2(psi_a, psi_b,
                                         obj_dml_oneway_cluster_data.cluster_vars[:, 0],
                                         smpls_one_split)
    else:
        theta = res_manual['theta']
    psi = np.multiply(u_hat - w_hat * theta, v_hat)
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
    assert math.isclose(dml_pliv_oneway_cluster_fixture['coef'],
                        dml_pliv_oneway_cluster_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_pliv_oneway_cluster_se(dml_pliv_oneway_cluster_fixture):
    assert math.isclose(dml_pliv_oneway_cluster_fixture['se'],
                        dml_pliv_oneway_cluster_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.fixture(scope="module")
def dml_plr_cluster_with_index(generate_data1, learner, dml_procedure):
    # in the one-way cluster case with exactly one observation per cluster, we get the same result w & w/o clustering
    n_folds = 2

    # collect data
    data = generate_data1
    x_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for m & g
    ml_g = clone(learner)
    ml_m = clone(learner)

    obj_dml_data = dml.DoubleMLData(data, 'y', ['d'], x_cols)
    np.random.seed(3141)
    dml_plr_obj = dml.DoubleMLPLR(obj_dml_data,
                                  ml_g, ml_m,
                                  n_folds,
                                  dml_procedure=dml_procedure)
    dml_plr_obj.fit()

    df = data.reset_index()
    dml_cluster_data = dml.DoubleMLClusterData(df,
                                               y_col='y',
                                               d_cols='d',
                                               x_cols=x_cols,
                                               cluster_cols='index')
    np.random.seed(3141)
    dml_plr_cluster_obj = dml.DoubleMLPLR(dml_cluster_data,
                                          ml_g, ml_m,
                                          n_folds,
                                          dml_procedure=dml_procedure)
    dml_plr_cluster_obj.fit()

    res_dict = {'coef': dml_plr_obj.coef,
                'coef_manual': dml_plr_cluster_obj.coef,
                'se': dml_plr_obj.se,
                'se_manual': dml_plr_cluster_obj.se}

    return res_dict


@pytest.mark.ci
def test_dml_plr_cluster_with_index_coef(dml_plr_cluster_with_index):
    assert math.isclose(dml_plr_cluster_with_index['coef'],
                        dml_plr_cluster_with_index['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_plr_cluster_with_index_se(dml_plr_cluster_with_index):
    assert math.isclose(dml_plr_cluster_with_index['se'],
                        dml_plr_cluster_with_index['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)
