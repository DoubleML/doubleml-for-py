import numpy as np
import pytest
import math

from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

import doubleml as dml

from doubleml.tests.helper_general import get_n_datasets
from doubleml.tests.helper_pliv_partial_z_manual import pliv_partial_z_dml1, pliv_partial_z_dml2, \
    fit_nuisance_pliv_partial_z, boot_pliv_partial_z


# number of datasets per dgp
n_datasets = get_n_datasets()


@pytest.fixture(scope='module',
                params=range(n_datasets))
def idx(request):
    return request.param


@pytest.fixture(scope='module',
                params=[Lasso(alpha=0.1)])
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
def dml_pliv_partial_z_fixture(generate_data_pliv_partialZ, idx, learner, score, dml_procedure):
    boot_methods = ['Bayes', 'normal', 'wild']
    n_folds = 2
    n_rep_boot = 503

    # collect data
    data = generate_data_pliv_partialZ[idx]
    X_cols = data.columns[data.columns.str.startswith('X')].tolist()
    Z_cols = data.columns[data.columns.str.startswith('Z')].tolist()

    # Set machine learning methods for r
    ml_r = clone(learner)

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData(data, 'y', ['d'], X_cols, Z_cols)
    dml_pliv_obj = dml.DoubleMLPLIV._partialZ(obj_dml_data,
                                             ml_r,
                                             n_folds,
                                             dml_procedure=dml_procedure)

    dml_pliv_obj.fit()
    
    np.random.seed(3141)
    y = data['y'].values
    X = data.loc[:, X_cols].values
    d = data['d'].values
    z = data.loc[:, Z_cols].values
    resampling = KFold(n_splits=n_folds,
                       shuffle=True)
    smpls = [(train, test) for train, test in resampling.split(X)]
    
    r_hat = fit_nuisance_pliv_partial_z(y, X, d, z,
                                        clone(learner),
                                        smpls)
    
    if dml_procedure == 'dml1':
        res_manual, se_manual = pliv_partial_z_dml1(y, X, d,
                                                    z,
                                                    r_hat,
                                                    smpls, score)
    elif dml_procedure == 'dml2':
        res_manual, se_manual = pliv_partial_z_dml2(y, X, d,
                                                    z,
                                                    r_hat,
                                                    smpls, score)
    
    res_dict = {'coef': dml_pliv_obj.coef,
                'coef_manual': res_manual,
                'se': dml_pliv_obj.se,
                'se_manual': se_manual,
                'boot_methods': boot_methods}
    
    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_theta, boot_t_stat = boot_pliv_partial_z(res_manual,
                                                      y, d,
                                                      z,
                                                      r_hat,
                                                      smpls, score,
                                                      se_manual,
                                                      bootstrap, n_rep_boot,
                                                      dml_procedure)
        
        np.random.seed(3141)
        dml_pliv_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_coef' + bootstrap] = dml_pliv_obj.boot_coef
        res_dict['boot_t_stat' + bootstrap] = dml_pliv_obj.boot_t_stat
        res_dict['boot_coef' + bootstrap + '_manual'] = boot_theta
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat
    
    return res_dict


def test_dml_pliv_coef(dml_pliv_partial_z_fixture):
    assert math.isclose(dml_pliv_partial_z_fixture['coef'],
                        dml_pliv_partial_z_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_pliv_se(dml_pliv_partial_z_fixture):
    assert math.isclose(dml_pliv_partial_z_fixture['se'],
                        dml_pliv_partial_z_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_pliv_boot(dml_pliv_partial_z_fixture):
    for bootstrap in dml_pliv_partial_z_fixture['boot_methods']:
        assert np.allclose(dml_pliv_partial_z_fixture['boot_coef' + bootstrap],
                           dml_pliv_partial_z_fixture['boot_coef' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_pliv_partial_z_fixture['boot_t_stat' + bootstrap],
                           dml_pliv_partial_z_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)

