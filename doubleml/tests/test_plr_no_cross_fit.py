import numpy as np
import pytest
import math
import scipy

from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

import doubleml as dml

from doubleml.tests.helper_general import get_n_datasets
from doubleml.tests.helper_plr_manual import plr_dml1, plr_dml2, fit_nuisance_plr, boot_plr


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
                params=[1, 2])
def n_folds(request):
    return request.param


@pytest.fixture(scope="module")
def dml_plr_no_cross_fit_fixture(generate_data1, idx, learner, score, n_folds):
    boot_methods = ['normal']
    n_rep_boot = 502
    dml_procedure = 'dml1'

    # collect data
    data = generate_data1[idx]
    X_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for m & g
    ml_g = clone(learner)
    ml_m = clone(learner)

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData(data, 'y', ['d'], X_cols)
    dml_plr_obj = dml.DoubleMLPLR(obj_dml_data,
                                  ml_g, ml_m,
                                  n_folds,
                                  score=score,
                                  dml_procedure=dml_procedure,
                                  apply_cross_fitting=False)

    dml_plr_obj.fit()
    
    np.random.seed(3141)
    y = data['y'].values
    X = data.loc[:, X_cols].values
    d = data['d'].values
    if n_folds == 1:
        smpls = [(np.arange(len(y)), np.arange(len(y)))]
    else:
        resampling = KFold(n_splits=n_folds,
                           shuffle=True)
        smpls = [(train, test) for train, test in resampling.split(X)]
        smpls = [smpls[0]]
    
    g_hat, m_hat = fit_nuisance_plr(y, X, d,
                                    clone(learner), clone(learner), smpls)

    assert dml_procedure == 'dml1'
    res_manual, se_manual = plr_dml1(y, X, d,
                                     g_hat, m_hat,
                                     smpls, score)
    
    res_dict = {'coef': dml_plr_obj.coef,
                'coef_manual': res_manual,
                'se': dml_plr_obj.se,
                'se_manual': se_manual,
                'boot_methods': boot_methods}
    
    #for bootstrap in boot_methods:
    #    np.random.seed(3141)
    #    boot_theta = boot_plr(res_manual,
    #                          y, d,
    #                          g_hat, m_hat,
    #                          smpls, score,
    #                          se_manual,
    #                          bootstrap, n_rep_boot,
    #                          dml_procedure)
    #
    #    np.random.seed(3141)
    #    dml_plr_obj.bootstrap(method = bootstrap, n_rep=n_rep_boot)
    #    res_dict['boot_coef' + bootstrap] = dml_plr_obj.boot_coef
    #    res_dict['boot_coef' + bootstrap + '_manual'] = boot_theta
    
    return res_dict


def test_dml_plr_coef(dml_plr_no_cross_fit_fixture):
    assert math.isclose(dml_plr_no_cross_fit_fixture['coef'],
                        dml_plr_no_cross_fit_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_plr_se(dml_plr_no_cross_fit_fixture):
    assert math.isclose(dml_plr_no_cross_fit_fixture['se'],
                        dml_plr_no_cross_fit_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


#def test_dml_plr_boot(dml_plr_no_cross_fit_fixture):
#    for bootstrap in dml_plr_no_cross_fit_fixture['boot_methods']:
#        assert np.allclose(dml_plr_no_cross_fit_fixture['boot_coef' + bootstrap],
#                           dml_plr_no_cross_fit_fixture['boot_coef' + bootstrap + '_manual'],
#                           rtol=1e-9, atol=1e-4)


@pytest.fixture(scope='module',
                params=[10, 13])
def n_rep_cross_fit(request):
    return request.param


@pytest.fixture(scope="module")
def dml_plr_rep_no_cross_fit_fixture(generate_data1, idx, learner, score, n_rep_cross_fit):
    boot_methods = ['normal']
    n_folds = 2
    n_rep_boot = 498
    dml_procedure = 'dml1'

    # collect data
    data = generate_data1[idx]
    X_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for m & g
    ml_g = clone(learner)
    ml_m = clone(learner)

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData(data, 'y', ['d'], X_cols)
    dml_plr_obj = dml.DoubleMLPLR(obj_dml_data,
                                  ml_g, ml_m,
                                  n_folds,
                                  n_rep_cross_fit,
                                  score,
                                  dml_procedure,
                                  apply_cross_fitting=False)

    dml_plr_obj.fit()

    np.random.seed(3141)
    y = data['y'].values
    X = data.loc[:, X_cols].values
    d = data['d'].values
    all_smpls = []
    for i_rep in range(n_rep_cross_fit):
        resampling = KFold(n_splits=n_folds,
                           shuffle=True)
        smpls = [(train, test) for train, test in resampling.split(X)]
        all_smpls.append(smpls)

    # adapt to do no-cross-fitting in each repetition
    all_smpls = [[xx[0]] for xx in all_smpls]

    thetas = np.zeros(n_rep_cross_fit)
    ses = np.zeros(n_rep_cross_fit)
    all_g_hat = list()
    all_m_hat = list()
    for i_rep in range(n_rep_cross_fit):
        smpls = all_smpls[i_rep]

        g_hat, m_hat = fit_nuisance_plr(y, X, d,
                                        clone(learner), clone(learner), smpls)

        all_g_hat.append(g_hat)
        all_m_hat.append(m_hat)

        thetas[i_rep], ses[i_rep] = plr_dml1(y, X, d,
                                             all_g_hat[i_rep], all_m_hat[i_rep],
                                             smpls, score)

    res_manual = np.median(thetas)
    se_manual = np.sqrt(np.median(np.power(ses, 2) - np.power(thetas - res_manual, 2)))

    res_dict = {'coef': dml_plr_obj.coef,
                'coef_manual': res_manual,
                'se': dml_plr_obj.se,
                'se_manual': se_manual,
                'boot_methods': boot_methods
                }

    #for bootstrap in boot_methods:
    #    np.random.seed(3141)
    #    all_boot_theta = list()
    #    for i_rep in range(n_rep_cross_fit):
    #        smpls = all_smpls[i_rep]
    #        boot_theta = boot_plr(thetas[i_rep],
    #                              y, d,
    #                              all_g_hat[i_rep], all_m_hat[i_rep],
    #                              smpls, score,
    #                              ses[i_rep],
    #                              bootstrap, n_rep_boot,
    #                              dml_procedure)
    #        all_boot_theta.append(boot_theta)
    #
    #    boot_theta = np.hstack(all_boot_theta)
    #
    #    np.random.seed(3141)
    #    dml_plr_obj.bootstrap(method=bootstrap, n_rep=n_rep_boot)
    #    res_dict['boot_coef' + bootstrap] = dml_plr_obj.boot_coef
    #    res_dict['boot_coef' + bootstrap + '_manual'] = boot_theta

    return res_dict


def test_dml_plr_coef(dml_plr_rep_no_cross_fit_fixture):
    assert math.isclose(dml_plr_rep_no_cross_fit_fixture['coef'],
                        dml_plr_rep_no_cross_fit_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_plr_se(dml_plr_rep_no_cross_fit_fixture):
    assert math.isclose(dml_plr_rep_no_cross_fit_fixture['se'],
                        dml_plr_rep_no_cross_fit_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


#def test_dml_plr_boot(dml_plr_rep_no_cross_fit_fixture):
#    for bootstrap in dml_plr_rep_no_cross_fit_fixture['boot_methods']:
#        assert np.allclose(dml_plr_rep_no_cross_fit_fixture['boot_coef' + bootstrap],
#                           dml_plr_rep_no_cross_fit_fixture['boot_coef' + bootstrap + '_manual'],
#                           rtol=1e-9, atol=1e-4)
