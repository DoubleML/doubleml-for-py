import numpy as np
import pytest
import math

from sklearn.base import clone

from sklearn.linear_model import Lasso

import doubleml as dml

from ._utils import draw_smpls
from ._utils_plr_manual import fit_plr, plr_dml1, fit_nuisance_plr, boot_plr, tune_nuisance_plr


@pytest.fixture(scope='module',
                params=[Lasso(alpha=0.1)])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['IV-type', 'partialling out'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=[1, 2])
def n_folds(request):
    return request.param


@pytest.fixture(scope="module")
def dml_plr_no_cross_fit_fixture(generate_data1, learner, score, n_folds):
    boot_methods = ['normal']
    n_rep_boot = 502
    dml_procedure = 'dml1'

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
                                  n_folds,
                                  score=score,
                                  dml_procedure=dml_procedure,
                                  apply_cross_fitting=False)

    dml_plr_obj.fit()

    np.random.seed(3141)
    y = data['y'].values
    x = data.loc[:, x_cols].values
    d = data['d'].values
    if n_folds == 1:
        smpls = [(np.arange(len(y)), np.arange(len(y)))]
    else:
        n_obs = len(y)
        all_smpls = draw_smpls(n_obs, n_folds)
        smpls = all_smpls[0]
        smpls = [smpls[0]]

    res_manual = fit_plr(y, x, d, clone(learner), clone(learner),
                         [smpls], dml_procedure, score)

    res_dict = {'coef': dml_plr_obj.coef,
                'coef_manual': res_manual['theta'],
                'se': dml_plr_obj.se,
                'se_manual': res_manual['se'],
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_theta, boot_t_stat = boot_plr(y, d, res_manual['thetas'], res_manual['ses'],
                                           res_manual['all_g_hat'], res_manual['all_m_hat'],
                                           [smpls], score, bootstrap, n_rep_boot,
                                           apply_cross_fitting=False)

        np.random.seed(3141)
        dml_plr_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_coef' + bootstrap] = dml_plr_obj.boot_coef
        res_dict['boot_t_stat' + bootstrap] = dml_plr_obj.boot_t_stat
        res_dict['boot_coef' + bootstrap + '_manual'] = boot_theta
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat

    return res_dict


@pytest.mark.ci
def test_dml_plr_no_cross_fit_coef(dml_plr_no_cross_fit_fixture):
    assert math.isclose(dml_plr_no_cross_fit_fixture['coef'],
                        dml_plr_no_cross_fit_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_plr_no_cross_fit_se(dml_plr_no_cross_fit_fixture):
    assert math.isclose(dml_plr_no_cross_fit_fixture['se'],
                        dml_plr_no_cross_fit_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_plr_no_cross_fit_boot(dml_plr_no_cross_fit_fixture):
    for bootstrap in dml_plr_no_cross_fit_fixture['boot_methods']:
        assert np.allclose(dml_plr_no_cross_fit_fixture['boot_coef' + bootstrap],
                           dml_plr_no_cross_fit_fixture['boot_coef' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)


@pytest.fixture(scope='module',
                params=[10, 13])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module")
def dml_plr_rep_no_cross_fit_fixture(generate_data1, learner, score, n_rep):
    boot_methods = ['normal']
    n_folds = 2
    n_rep_boot = 498
    dml_procedure = 'dml1'

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
                                  n_folds,
                                  n_rep,
                                  score,
                                  dml_procedure,
                                  apply_cross_fitting=False)

    dml_plr_obj.fit()

    np.random.seed(3141)
    y = data['y'].values
    x = data.loc[:, x_cols].values
    d = data['d'].values
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds, n_rep)

    # adapt to do no-cross-fitting in each repetition
    all_smpls = [[xx[0]] for xx in all_smpls]

    thetas = np.zeros(n_rep)
    ses = np.zeros(n_rep)
    all_g_hat = list()
    all_m_hat = list()
    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]

        g_hat, m_hat = fit_nuisance_plr(y, x, d,
                                        clone(learner), clone(learner), smpls)

        all_g_hat.append(g_hat)
        all_m_hat.append(m_hat)

        thetas[i_rep], ses[i_rep] = plr_dml1(y, x, d,
                                             all_g_hat[i_rep], all_m_hat[i_rep],
                                             smpls, score)

    res_manual = np.median(thetas)
    se_manual = np.sqrt(np.median(np.power(ses, 2)*len(smpls[0][1]) + np.power(thetas - res_manual, 2))/len(smpls[0][1]))

    res_dict = {'coef': dml_plr_obj.coef,
                'coef_manual': res_manual,
                'se': dml_plr_obj.se,
                'se_manual': se_manual,
                'boot_methods': boot_methods
                }

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_theta, boot_t_stat = boot_plr(y, d, thetas, ses,
                                           all_g_hat, all_m_hat,
                                           all_smpls, score, bootstrap, n_rep_boot,
                                           n_rep=n_rep, apply_cross_fitting=False)

        np.random.seed(3141)
        dml_plr_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_coef' + bootstrap] = dml_plr_obj.boot_coef
        res_dict['boot_t_stat' + bootstrap] = dml_plr_obj.boot_t_stat
        res_dict['boot_coef' + bootstrap + '_manual'] = boot_theta
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat

    return res_dict


@pytest.mark.ci
def test_dml_plr_rep_no_cross_fit_coef(dml_plr_rep_no_cross_fit_fixture):
    assert math.isclose(dml_plr_rep_no_cross_fit_fixture['coef'],
                        dml_plr_rep_no_cross_fit_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_plr_rep_no_cross_fit_se(dml_plr_rep_no_cross_fit_fixture):
    assert math.isclose(dml_plr_rep_no_cross_fit_fixture['se'],
                        dml_plr_rep_no_cross_fit_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_plr_rep_no_cross_fit_boot(dml_plr_rep_no_cross_fit_fixture):
    for bootstrap in dml_plr_rep_no_cross_fit_fixture['boot_methods']:
        assert np.allclose(dml_plr_rep_no_cross_fit_fixture['boot_coef' + bootstrap],
                           dml_plr_rep_no_cross_fit_fixture['boot_coef' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_plr_rep_no_cross_fit_fixture['boot_t_stat' + bootstrap],
                           dml_plr_rep_no_cross_fit_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)


@pytest.fixture(scope='module',
                params=[True, False])
def tune_on_folds(request):
    return request.param


@pytest.fixture(scope="module")
def dml_plr_no_cross_fit_tune_fixture(generate_data1, learner, score, tune_on_folds):
    par_grid = {'ml_g': {'alpha': np.linspace(0.05, .95, 7)},
                'ml_m': {'alpha': np.linspace(0.05, .95, 7)}}
    n_folds_tune = 3

    boot_methods = ['normal']
    n_rep_boot = 502
    dml_procedure = 'dml1'

    # collect data
    data = generate_data1
    x_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for m & g
    ml_g = Lasso()
    ml_m = Lasso()

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData(data, 'y', ['d'], x_cols)
    dml_plr_obj = dml.DoubleMLPLR(obj_dml_data,
                                  ml_g, ml_m,
                                  n_folds=2,
                                  score=score,
                                  dml_procedure=dml_procedure,
                                  apply_cross_fitting=False)

    # tune hyperparameters
    _ = dml_plr_obj.tune(par_grid, tune_on_folds=tune_on_folds, n_folds_tune=n_folds_tune)

    # fit with tuned parameters
    dml_plr_obj.fit()

    np.random.seed(3141)
    y = obj_dml_data.y
    x = obj_dml_data.x
    d = obj_dml_data.d
    n_obs = len(y)

    all_smpls = draw_smpls(n_obs, 2)
    smpls = all_smpls[0]
    smpls = [smpls[0]]

    if tune_on_folds:
        g_params, m_params = tune_nuisance_plr(y, x, d,
                                               clone(ml_g), clone(ml_m), smpls, n_folds_tune,
                                               par_grid['ml_g'], par_grid['ml_m'])
    else:
        xx = [(np.arange(len(y)), np.array([]))]
        g_params, m_params = tune_nuisance_plr(y, x, d,
                                               clone(ml_g), clone(ml_m), xx, n_folds_tune,
                                               par_grid['ml_g'], par_grid['ml_m'])

    res_manual = fit_plr(y, x, d, clone(ml_m), clone(ml_g),
                         [smpls], dml_procedure, score, g_params=g_params, m_params=m_params)

    res_dict = {'coef': dml_plr_obj.coef,
                'coef_manual': res_manual['theta'],
                'se': dml_plr_obj.se,
                'se_manual': res_manual['se'],
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_theta, boot_t_stat = boot_plr(y, d, res_manual['thetas'], res_manual['ses'],
                                           res_manual['all_g_hat'], res_manual['all_m_hat'],
                                           [smpls], score, bootstrap, n_rep_boot,
                                           apply_cross_fitting=False)

        np.random.seed(3141)
        dml_plr_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_coef' + bootstrap] = dml_plr_obj.boot_coef
        res_dict['boot_t_stat' + bootstrap] = dml_plr_obj.boot_t_stat
        res_dict['boot_coef' + bootstrap + '_manual'] = boot_theta
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat

    return res_dict


@pytest.mark.ci
def test_dml_plr_no_cross_fit_tune_coef(dml_plr_no_cross_fit_tune_fixture):
    assert math.isclose(dml_plr_no_cross_fit_tune_fixture['coef'],
                        dml_plr_no_cross_fit_tune_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_plr_no_cross_fit_tune_se(dml_plr_no_cross_fit_tune_fixture):
    assert math.isclose(dml_plr_no_cross_fit_tune_fixture['se'],
                        dml_plr_no_cross_fit_tune_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_plr_no_cross_fit_tune_boot(dml_plr_no_cross_fit_tune_fixture):
    for bootstrap in dml_plr_no_cross_fit_tune_fixture['boot_methods']:
        assert np.allclose(dml_plr_no_cross_fit_tune_fixture['boot_coef' + bootstrap],
                           dml_plr_no_cross_fit_tune_fixture['boot_coef' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_plr_no_cross_fit_tune_fixture['boot_t_stat' + bootstrap],
                           dml_plr_no_cross_fit_tune_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
