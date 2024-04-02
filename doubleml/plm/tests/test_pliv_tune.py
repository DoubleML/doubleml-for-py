import numpy as np
import pytest
import math

from sklearn.linear_model import Lasso, ElasticNet

import doubleml as dml

from ...tests._utils import draw_smpls, _clone
from ._utils_pliv_manual import fit_pliv, boot_pliv, tune_nuisance_pliv


@pytest.fixture(scope='module',
                params=[Lasso(),
                        ElasticNet()])
def learner_l(request):
    return request.param


@pytest.fixture(scope='module',
                params=[ElasticNet()])
def learner_m(request):
    return request.param


@pytest.fixture(scope='module',
                params=[ElasticNet()])
def learner_r(request):
    return request.param


@pytest.fixture(scope='module',
                params=[ElasticNet()])
def learner_g(request):
    return request.param


@pytest.fixture(scope='module',
                params=['partialling out', 'IV-type'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=[True, False])
def tune_on_folds(request):
    return request.param


def get_par_grid(learner):
    if learner.__class__ == Lasso:
        par_grid = {'alpha': np.linspace(0.05, .95, 7)}
    else:
        assert learner.__class__ == ElasticNet
        par_grid = {'l1_ratio': [.1, .5, .7, .9, .95, .99, 1], 'alpha': np.linspace(0.05, 1., 7)}
    return par_grid


@pytest.fixture(scope='module')
def dml_pliv_fixture(generate_data_iv, learner_l, learner_m, learner_r, learner_g, score, tune_on_folds):
    par_grid = {'ml_l': get_par_grid(learner_l),
                'ml_m': get_par_grid(learner_m),
                'ml_r': get_par_grid(learner_r),
                'ml_g': get_par_grid(learner_g)}
    n_folds_tune = 4

    boot_methods = ['Bayes', 'normal', 'wild']
    n_folds = 2
    n_rep_boot = 503

    # collect data
    data = generate_data_iv
    x_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for l, m, r & g
    ml_l = _clone(learner_l)
    ml_m = _clone(learner_m)
    ml_r = _clone(learner_r)
    if score == 'IV-type':
        ml_g = _clone(learner_g)
    else:
        ml_g = None

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData(data, 'y', ['d'], x_cols, 'Z1')
    dml_pliv_obj = dml.DoubleMLPLIV(obj_dml_data,
                                    ml_l, ml_m, ml_r, ml_g,
                                    n_folds=n_folds,
                                    score=score)

    # tune hyperparameters
    tune_res = dml_pliv_obj.tune(par_grid, tune_on_folds=tune_on_folds, n_folds_tune=n_folds_tune,
                                 return_tune_res=False)
    assert isinstance(tune_res, dml.DoubleMLPLIV)

    dml_pliv_obj.fit()

    np.random.seed(3141)
    y = data['y'].values
    x = data.loc[:, x_cols].values
    d = data['d'].values
    z = data['Z1'].values
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds)
    smpls = all_smpls[0]

    tune_g = (score == 'IV-type') | callable(score)
    if tune_on_folds:
        l_params, m_params, r_params, g_params = tune_nuisance_pliv(
            y, x, d, z,
            _clone(learner_l), _clone(learner_m), _clone(learner_r), _clone(learner_g),
            smpls, n_folds_tune,
            par_grid['ml_l'], par_grid['ml_m'], par_grid['ml_r'], par_grid['ml_g'],
            tune_g)
    else:
        xx = [(np.arange(len(y)), np.array([]))]
        l_params, m_params, r_params, g_params = tune_nuisance_pliv(
            y, x, d, z,
            _clone(learner_l), _clone(learner_m), _clone(learner_r), _clone(learner_g),
            xx, n_folds_tune,
            par_grid['ml_l'], par_grid['ml_m'], par_grid['ml_r'], par_grid['ml_g'],
            tune_g)

        l_params = l_params * n_folds
        m_params = m_params * n_folds
        r_params = r_params * n_folds
        g_params = g_params * n_folds

    res_manual = fit_pliv(y, x, d, z, _clone(learner_l), _clone(learner_m), _clone(learner_r), _clone(learner_g),
                          all_smpls, score,
                          l_params=l_params, m_params=m_params, r_params=r_params, g_params=g_params)

    res_dict = {'coef': dml_pliv_obj.coef,
                'coef_manual': res_manual['theta'],
                'se': dml_pliv_obj.se,
                'se_manual': res_manual['se'],
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_t_stat = boot_pliv(y, d, z, res_manual['thetas'], res_manual['ses'],
                                res_manual['all_l_hat'], res_manual['all_m_hat'],
                                res_manual['all_r_hat'], res_manual['all_g_hat'],
                                all_smpls, score, bootstrap, n_rep_boot)

        np.random.seed(3141)
        dml_pliv_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_t_stat' + bootstrap] = dml_pliv_obj.boot_t_stat
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat.reshape(-1, 1, 1)

    return res_dict


@pytest.mark.ci
def test_dml_pliv_coef(dml_pliv_fixture):
    assert math.isclose(dml_pliv_fixture['coef'],
                        dml_pliv_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_pliv_se(dml_pliv_fixture):
    assert math.isclose(dml_pliv_fixture['se'],
                        dml_pliv_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_pliv_boot(dml_pliv_fixture):
    for bootstrap in dml_pliv_fixture['boot_methods']:
        assert np.allclose(dml_pliv_fixture['boot_t_stat' + bootstrap],
                           dml_pliv_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
