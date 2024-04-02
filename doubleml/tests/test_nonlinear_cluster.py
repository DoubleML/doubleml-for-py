import numpy as np
import pytest
import math

from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

import doubleml as dml
from doubleml.datasets import make_pliv_multiway_cluster_CKMS2021, DoubleMLClusterData

from .test_nonlinear_score_mixin import DoubleMLPLRWithNonLinearScoreMixin

np.random.seed(1234)
# Set the simulation parameters
N = 25  # number of observations (first dimension)
M = 25  # number of observations (second dimension)
dim_x = 100  # dimension of x

# create data without insturment for plr
x, y, d, cluster_vars, z = make_pliv_multiway_cluster_CKMS2021(N, M, dim_x, return_type="array")
obj_dml_cluster_data = DoubleMLClusterData.from_arrays(x, y, d, cluster_vars)

x, y, d, cluster_vars, z = make_pliv_multiway_cluster_CKMS2021(N, M, dim_x,
                                                               omega_X=np.array([0.25, 0]),
                                                               omega_epsilon=np.array([0.25, 0]),
                                                               omega_v=np.array([0.25, 0]),
                                                               omega_V=np.array([0.25, 0]),
                                                               return_type='array')
obj_dml_oneway_cluster_data = DoubleMLClusterData.from_arrays(x, y, d, cluster_vars)

# only the first cluster variable is relevant with the weight setting above
obj_dml_oneway_cluster_data.cluster_cols = 'cluster_var1'


@pytest.fixture(scope='module',
                params=[RandomForestRegressor(max_depth=2, n_estimators=10),
                        LinearRegression(),
                        Lasso(alpha=0.1)])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['IV-type', 'partialling out'])
def score(request):
    return request.param


@pytest.fixture(scope='module')
def dml_plr_oneway_cluster_linear_vs_nonlinear_fixture(learner, score):
    n_folds = 3

    # Set machine learning methods for l, m & g
    ml_l = clone(learner)
    ml_m = clone(learner)
    ml_g = clone(learner)

    np.random.seed(3141)
    if score == 'partialling out':
        dml_plr_obj = dml.DoubleMLPLR(obj_dml_oneway_cluster_data,
                                      ml_l, ml_m,
                                      n_folds=n_folds,
                                      score=score)
    else:
        assert score == 'IV-type'
        dml_plr_obj = dml.DoubleMLPLR(obj_dml_oneway_cluster_data,
                                      ml_l, ml_m, ml_g,
                                      n_folds=n_folds,
                                      score=score)

    np.random.seed(3141)
    dml_plr_obj.fit()

    np.random.seed(3141)
    if score == 'partialling out':
        dml_plr_obj2 = DoubleMLPLRWithNonLinearScoreMixin(obj_dml_oneway_cluster_data,
                                                          ml_l, ml_m,
                                                          n_folds=n_folds,
                                                          score=score)
    else:
        assert score == 'IV-type'
        dml_plr_obj2 = DoubleMLPLRWithNonLinearScoreMixin(obj_dml_oneway_cluster_data,
                                                          ml_l, ml_m, ml_g,
                                                          n_folds=n_folds,
                                                          score=score)

    np.random.seed(3141)
    dml_plr_obj2.fit()

    res_dict = {'coef_linear': dml_plr_obj.coef,
                'coef_nonlinear': dml_plr_obj2.coef,
                'se_linear': dml_plr_obj.se,
                'se_nonlinear': dml_plr_obj2.se}

    return res_dict


@pytest.mark.ci
def test_dml_plr_oneway_cluster_linear_vs_nonlinear_coef(dml_plr_oneway_cluster_linear_vs_nonlinear_fixture):
    assert math.isclose(dml_plr_oneway_cluster_linear_vs_nonlinear_fixture['coef_linear'][0],
                        dml_plr_oneway_cluster_linear_vs_nonlinear_fixture['coef_nonlinear'][0],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_plr_oneway_cluster_linear_vs_nonlinear_se(dml_plr_oneway_cluster_linear_vs_nonlinear_fixture):
    assert math.isclose(dml_plr_oneway_cluster_linear_vs_nonlinear_fixture['se_linear'][0],
                        dml_plr_oneway_cluster_linear_vs_nonlinear_fixture['se_nonlinear'][0],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.fixture(scope='module')
def dml_plr_multiway_cluster_linear_vs_nonlinear_fixture(learner, score):
    n_folds = 2
    n_rep = 2

    # Set machine learning methods for l, m & g
    ml_l = clone(learner)
    ml_m = clone(learner)
    ml_g = clone(learner)

    np.random.seed(3141)
    if score == 'partialling out':
        dml_plr_obj = dml.DoubleMLPLR(obj_dml_oneway_cluster_data,
                                      ml_l, ml_m,
                                      n_folds=n_folds,
                                      n_rep=n_rep,
                                      score=score)
    else:
        assert score == 'IV-type'
        dml_plr_obj = dml.DoubleMLPLR(obj_dml_oneway_cluster_data,
                                      ml_l, ml_m, ml_g,
                                      n_folds=n_folds,
                                      n_rep=n_rep,
                                      score=score)

    np.random.seed(3141)
    dml_plr_obj.fit()

    np.random.seed(3141)
    if score == 'partialling out':
        dml_plr_obj2 = DoubleMLPLRWithNonLinearScoreMixin(obj_dml_oneway_cluster_data,
                                                          ml_l, ml_m,
                                                          n_folds=n_folds,
                                                          n_rep=n_rep,
                                                          score=score)
    else:
        assert score == 'IV-type'
        dml_plr_obj2 = DoubleMLPLRWithNonLinearScoreMixin(obj_dml_oneway_cluster_data,
                                                          ml_l, ml_m, ml_g,
                                                          n_folds=n_folds,
                                                          n_rep=n_rep,
                                                          score=score)

    np.random.seed(3141)
    dml_plr_obj2.fit()

    res_dict = {'coef_linear': dml_plr_obj.coef,
                'coef_nonlinear': dml_plr_obj2.coef,
                'se_linear': dml_plr_obj.se,
                'se_nonlinear': dml_plr_obj2.se}

    return res_dict


@pytest.mark.ci
def test_dml_plr_multiway_cluster_linear_vs_nonlinear_coef(dml_plr_multiway_cluster_linear_vs_nonlinear_fixture):
    assert math.isclose(dml_plr_multiway_cluster_linear_vs_nonlinear_fixture['coef_linear'][0],
                        dml_plr_multiway_cluster_linear_vs_nonlinear_fixture['coef_nonlinear'][0],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_plr_multiway_cluster_linear_vs_nonlinear_se(dml_plr_multiway_cluster_linear_vs_nonlinear_fixture):
    assert math.isclose(dml_plr_multiway_cluster_linear_vs_nonlinear_fixture['se_linear'][0],
                        dml_plr_multiway_cluster_linear_vs_nonlinear_fixture['se_nonlinear'][0],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.fixture(scope="module")
def dml_plr_cluster_nonlinear_with_index(generate_data1, learner):
    # in the one-way cluster case with exactly one observation per cluster, we get the same result w & w/o clustering
    n_folds = 2

    # collect data
    data = generate_data1
    x_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for m & l
    ml_l = clone(learner)
    ml_m = clone(learner)

    obj_dml_data = dml.DoubleMLData(data, 'y', ['d'], x_cols)
    np.random.seed(3141)
    dml_plr_obj = DoubleMLPLRWithNonLinearScoreMixin(obj_dml_data,
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
    dml_plr_cluster_obj = DoubleMLPLRWithNonLinearScoreMixin(dml_cluster_data,
                                                             ml_l, ml_m,
                                                             n_folds=n_folds)
    dml_plr_cluster_obj.fit()

    res_dict = {'coef': dml_plr_obj.coef,
                'coef_cluster': dml_plr_cluster_obj.coef,
                'se': dml_plr_obj.se,
                'se_cluster': dml_plr_cluster_obj.se}

    return res_dict


@pytest.mark.ci
def test_dml_plr_cluster_nonlinear_with_index_coef(dml_plr_cluster_nonlinear_with_index):
    assert math.isclose(dml_plr_cluster_nonlinear_with_index['coef'][0],
                        dml_plr_cluster_nonlinear_with_index['coef_cluster'][0],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_plr_cluster_nonlinear_with_index_se(dml_plr_cluster_nonlinear_with_index):
    assert math.isclose(dml_plr_cluster_nonlinear_with_index['se'][0],
                        dml_plr_cluster_nonlinear_with_index['se_cluster'][0],
                        rel_tol=1e-9, abs_tol=1e-4)
