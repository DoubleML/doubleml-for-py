import pytest
import numpy as np
import copy

import doubleml as dml
from doubleml import DoubleMLIRM, DoubleMLData
from doubleml.datasets import make_irm_data
from sklearn.linear_model import LinearRegression, LogisticRegression

from ._utils_doubleml_sensitivity_manual import doubleml_sensitivity_manual, \
    doubleml_sensitivity_benchmark_manual


@pytest.fixture(scope="module", params=[["X1"], ["X2"], ["X3"]])
def benchmarking_set(request):
    return request.param


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


@pytest.fixture(scope="module")
def test_dml_benchmark_fixture(benchmarking_set, n_rep):
    random_state = 42
    x, y, d = make_irm_data(n_obs=50, dim_x=5, theta=0, return_type="np.array")

    classifier_class = LogisticRegression
    regressor_class = LinearRegression

    np.random.seed(3141)
    dml_data = DoubleMLData.from_arrays(x=x, y=y, d=d)
    x_list_long = copy.deepcopy(dml_data.x_cols)
    dml_int = DoubleMLIRM(dml_data,
                          ml_m=classifier_class(random_state=random_state),
                          ml_g=regressor_class(),
                          n_folds=2,
                          n_rep=n_rep)
    dml_int.fit(store_predictions=True)
    dml_int.sensitivity_analysis()
    dml_ext = copy.deepcopy(dml_int)
    df_bm = dml_int.sensitivity_benchmark(benchmarking_set=benchmarking_set)

    np.random.seed(3141)
    dml_data_short = DoubleMLData.from_arrays(x=x, y=y, d=d)
    dml_data_short.x_cols = [x for x in x_list_long if x not in benchmarking_set]
    dml_short = DoubleMLIRM(dml_data_short,
                            ml_m=classifier_class(random_state=random_state),
                            ml_g=regressor_class(),
                            n_folds=2,
                            n_rep=n_rep)
    dml_short.fit(store_predictions=True)
    fit_args = {"external_predictions": {"d": {"ml_m": dml_short.predictions["ml_m"][:, :, 0],
                                               "ml_g0": dml_short.predictions["ml_g0"][:, :, 0],
                                               "ml_g1": dml_short.predictions["ml_g1"][:, :, 0],
                                               }
                                         },
                }
    dml_ext.sensitivity_analysis()
    df_bm_ext = dml_ext.sensitivity_benchmark(benchmarking_set=benchmarking_set, fit_args=fit_args)

    res_dict = {"default_benchmark": df_bm,
                "external_benchmark": df_bm_ext}

    return res_dict


@pytest.mark.ci
def test_dml_sensitivity_external_predictions(test_dml_benchmark_fixture):
    assert np.allclose(test_dml_benchmark_fixture["default_benchmark"],
                       test_dml_benchmark_fixture["external_benchmark"],
                       rtol=1e-9,
                       atol=1e-4)
