import numpy as np
import pytest
import math

from sklearn.linear_model import LinearRegression

import doubleml as dml
from doubleml.datasets import make_pliv_multiway_cluster_CKMS2021
from ._utils_doubleml_sensitivity_manual import doubleml_sensitivity_benchmark_manual

np.random.seed(1234)
# Set the simulation parameters
N = 25  # number of observations (first dimension)
M = 25  # number of observations (second dimension)
dim_x = 10  # dimension of x


(x, y, d, cluster_vars, z) = make_pliv_multiway_cluster_CKMS2021(N, M, dim_x, return_type='array')
obj_dml_cluster_data = dml.DoubleMLClusterData.from_arrays(x, y, d, cluster_vars)

(x, y, d, cluster_vars, z) = make_pliv_multiway_cluster_CKMS2021(N, M, dim_x,
                                                                 omega_X=np.array([0.25, 0]),
                                                                 omega_epsilon=np.array([0.25, 0]),
                                                                 omega_v=np.array([0.25, 0]),
                                                                 omega_V=np.array([0.25, 0]),
                                                                 return_type='array')
obj_dml_oneway_cluster_data = dml.DoubleMLClusterData.from_arrays(x, y, d, cluster_vars)
# only the first cluster variable is relevant with the weight setting above
obj_dml_oneway_cluster_data.cluster_cols = 'cluster_var1'


@pytest.fixture(scope='module',
                params=['IV-type', 'partialling out'])
def score(request):
    return request.param


@pytest.fixture(scope='module')
def dml_plr_multiway_cluster_sensitivity_rho0(score):
    n_folds = 3
    cf_y = 0.03
    cf_d = 0.04
    level = 0.95

    # Set machine learning methods for l, m & r
    ml_l = LinearRegression()
    ml_m = LinearRegression()
    ml_g = LinearRegression()

    np.random.seed(3141)
    if score == 'partialling out':
        dml_plr_obj = dml.DoubleMLPLR(obj_dml_cluster_data,
                                      ml_l, ml_m,
                                      n_folds=n_folds,
                                      score=score)
    else:
        assert score == 'IV-type'
        dml_plr_obj = dml.DoubleMLPLR(obj_dml_cluster_data,
                                      ml_l, ml_m, ml_g,
                                      n_folds=n_folds,
                                      score=score)

    dml_plr_obj.fit()
    dml_plr_obj.sensitivity_analysis(cf_y=cf_y, cf_d=cf_d,
                                     rho=0.0, level=level, null_hypothesis=0.0)
    benchmark = dml_plr_obj.sensitivity_benchmark(benchmarking_set=["X1"])
    benchmark_manual = doubleml_sensitivity_benchmark_manual(dml_obj=dml_plr_obj,
                                                             benchmarking_set=["X1"])
    res_dict = {
        'coef': dml_plr_obj.coef,
        'se': dml_plr_obj.se,
        'sensitivity_params': dml_plr_obj.sensitivity_params,
        'benchmark': benchmark,
        'benchmark_manual': benchmark_manual
    }

    return res_dict


@pytest.mark.ci
def test_dml_plr_multiway_cluster_sensitivity_coef(dml_plr_multiway_cluster_sensitivity_rho0):
    assert math.isclose(dml_plr_multiway_cluster_sensitivity_rho0['coef'][0],
                        dml_plr_multiway_cluster_sensitivity_rho0['sensitivity_params']['theta']['lower'][0],
                        rel_tol=1e-9, abs_tol=1e-4)
    assert math.isclose(dml_plr_multiway_cluster_sensitivity_rho0['coef'][0],
                        dml_plr_multiway_cluster_sensitivity_rho0['sensitivity_params']['theta']['upper'][0],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_sensitivity_benchmark(dml_plr_multiway_cluster_sensitivity_rho0):
    expected_columns = ["cf_y", "cf_d", "rho", "delta_theta"]
    assert all(dml_plr_multiway_cluster_sensitivity_rho0['benchmark'].columns == expected_columns)
    assert all(dml_plr_multiway_cluster_sensitivity_rho0['benchmark'].index == ["d"])
    assert dml_plr_multiway_cluster_sensitivity_rho0['benchmark'].equals(
        dml_plr_multiway_cluster_sensitivity_rho0['benchmark_manual'])


@pytest.fixture(scope='module')
def dml_plr_multiway_cluster_sensitivity_rho0_se():
    n_folds = 3
    cf_y = 0.03
    cf_d = 0.04
    level = 0.95

    # Set machine learning methods for l, m & r
    ml_l = LinearRegression()
    ml_m = LinearRegression()

    np.random.seed(3141)
    dml_plr_obj = dml.DoubleMLPLR(obj_dml_cluster_data,
                                  ml_l, ml_m,
                                  n_folds=n_folds,
                                  score='partialling out')

    dml_plr_obj.fit()
    dml_plr_obj.sensitivity_analysis(cf_y=cf_y, cf_d=cf_d,
                                     rho=0.0, level=level, null_hypothesis=0.0)

    res_dict = {'coef': dml_plr_obj.coef,
                'se': dml_plr_obj.se,
                'sensitivity_params': dml_plr_obj.sensitivity_params}

    return res_dict


# only valid for 'partialling out '; This might have slightly less precision in the calculations
@pytest.mark.ci
def test_dml_pliv_multiway_cluster_sensitivity_se(dml_plr_multiway_cluster_sensitivity_rho0_se):
    assert math.isclose(dml_plr_multiway_cluster_sensitivity_rho0_se['se'][0],
                        dml_plr_multiway_cluster_sensitivity_rho0_se['sensitivity_params']['se']['lower'][0],
                        rel_tol=1e-9, abs_tol=1e-3)
    assert math.isclose(dml_plr_multiway_cluster_sensitivity_rho0_se['se'][0],
                        dml_plr_multiway_cluster_sensitivity_rho0_se['sensitivity_params']['se']['upper'][0],
                        rel_tol=1e-9, abs_tol=1e-3)
