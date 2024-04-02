import pytest
import numpy as np

import doubleml as dml
from sklearn.linear_model import LinearRegression

from ._utils_doubleml_sensitivity_manual import doubleml_sensitivity_manual, \
    doubleml_sensitivity_benchmark_manual


@pytest.fixture(scope='module',
                params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope='module',
                params=[0.03, 0.3])
def cf_y(request):
    return request.param


@pytest.fixture(scope='module',
                params=[0.03, 0.3])
def cf_d(request):
    return request.param


@pytest.fixture(scope='module',
                params=[-0.5, 0.0, 1.0])
def rho(request):
    return request.param


@pytest.fixture(scope='module',
                params=[0.8, 0.95])
def level(request):
    return request.param


@pytest.fixture(scope="module")
def dml_sensitivity_multitreat_fixture(generate_data_bivariate, n_rep, cf_y, cf_d, rho, level):

    # collect data
    data = generate_data_bivariate
    x_cols = data.columns[data.columns.str.startswith('X')].tolist()
    d_cols = data.columns[data.columns.str.startswith('d')].tolist()

    # Set machine learning methods for m & g
    ml_l = LinearRegression()
    ml_m = LinearRegression()

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData(data, 'y', d_cols, x_cols)
    dml_plr_obj = dml.DoubleMLPLR(obj_dml_data,
                                  ml_l,
                                  ml_m,
                                  n_folds=5,
                                  n_rep=n_rep,
                                  score='partialling out')

    dml_plr_obj.fit()
    dml_plr_obj.sensitivity_analysis(cf_y=cf_y, cf_d=cf_d, rho=rho, level=level, null_hypothesis=0.0)
    res_manual = doubleml_sensitivity_manual(sensitivity_elements=dml_plr_obj.sensitivity_elements,
                                             all_coefs=dml_plr_obj.all_coef,
                                             psi=dml_plr_obj.psi,
                                             psi_deriv=dml_plr_obj.psi_deriv,
                                             cf_y=cf_y,
                                             cf_d=cf_d,
                                             rho=rho,
                                             level=level)
    benchmark = dml_plr_obj.sensitivity_benchmark(benchmarking_set=["X1"])

    benchmark_manual = doubleml_sensitivity_benchmark_manual(dml_obj=dml_plr_obj,
                                                             benchmarking_set=["X1"])
    res_dict = {'sensitivity_params': dml_plr_obj.sensitivity_params,
                'sensitivity_params_manual': res_manual,
                'benchmark': benchmark,
                'benchmark_manual': benchmark_manual,
                'd_cols': d_cols,
                }

    return res_dict


@pytest.mark.ci
def test_dml_sensitivity_params(dml_sensitivity_multitreat_fixture):
    sensitivity_param_names = ['theta', 'se', 'ci']
    for sensitivity_param in sensitivity_param_names:
        for bound in ['lower', 'upper']:
            assert np.allclose(dml_sensitivity_multitreat_fixture['sensitivity_params'][sensitivity_param][bound],
                               dml_sensitivity_multitreat_fixture['sensitivity_params_manual'][sensitivity_param][bound])


@pytest.mark.ci
def test_dml_sensitivity_benchmark(dml_sensitivity_multitreat_fixture):
    expected_columns = ["cf_y", "cf_d", "rho", "delta_theta"]
    assert all(dml_sensitivity_multitreat_fixture['benchmark'].columns == expected_columns)
    assert all(dml_sensitivity_multitreat_fixture['benchmark'].index ==
               dml_sensitivity_multitreat_fixture['d_cols'])
    assert dml_sensitivity_multitreat_fixture['benchmark'].equals(dml_sensitivity_multitreat_fixture['benchmark_manual'])
