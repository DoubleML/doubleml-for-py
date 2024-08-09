import numpy as np
import pytest
import math

from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

import doubleml as dml

from ...tests._utils import draw_smpls
from ._utils_apo_manual import fit_apo, boot_apo, tune_nuisance_apo


@pytest.fixture(scope='module',
                params=[RandomForestRegressor(random_state=42)])
def learner_g(request):
    return request.param


@pytest.fixture(scope='module',
                params=[LogisticRegression(random_state=42)])
def learner_m(request):
    return request.param


@pytest.fixture(scope='module',
                params=['APO'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=[True, False])
def normalize_ipw(request):
    return request.param


@pytest.fixture(scope='module',
                params=[True, False])
def tune_on_folds(request):
    return request.param


def get_par_grid(learner):
    if learner.__class__ in [RandomForestRegressor]:
        par_grid = {'n_estimators': [5, 10, 20]}
    else:
        assert learner.__class__ in [LogisticRegression]
        par_grid = {'C': np.logspace(-4, 2, 10)}
    return par_grid


@pytest.fixture(scope='module')
def dml_apo_tune_fixture(generate_data_irm, learner_g, learner_m, score, normalize_ipw, tune_on_folds):
    par_grid = {'ml_g': get_par_grid(learner_g),
                'ml_m': get_par_grid(learner_m)}
    n_folds_tune = 4

    boot_methods = ['normal']
    n_folds = 2
    n_rep_boot = 499
    treatment_level = 0

    # collect data
    (x, y, d) = generate_data_irm
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds, n_rep=1, groups=d)

    # Set machine learning methods for m & g
    ml_g = clone(learner_g)
    ml_m = clone(learner_m)

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d)
    dml_obj = dml.DoubleMLAPO(obj_dml_data,
                              ml_g, ml_m,
                              treatment_level=treatment_level,
                              n_folds=n_folds,
                              score=score,
                              normalize_ipw=normalize_ipw,
                              draw_sample_splitting=False)
    # synchronize the sample splitting
    dml_obj.set_sample_splitting(all_smpls=all_smpls)
    np.random.seed(3141)
    # tune hyperparameters
    tune_res = dml_obj.tune(par_grid, tune_on_folds=tune_on_folds, n_folds_tune=n_folds_tune,
                            return_tune_res=False)
    assert isinstance(tune_res, dml.DoubleMLAPO)

    dml_obj.fit()

    np.random.seed(3141)
    smpls = all_smpls[0]

    if tune_on_folds:
        g0_params, g1_params, m_params = tune_nuisance_apo(y, x, d, treatment_level,
                                                           clone(learner_g), clone(learner_m), smpls, score,
                                                           n_folds_tune,
                                                           par_grid['ml_g'], par_grid['ml_m'])
    else:
        xx = [(np.arange(len(y)), np.array([]))]
        g0_params, g1_params, m_params = tune_nuisance_apo(y, x, d, treatment_level,
                                                           clone(learner_g), clone(learner_m), xx, score,
                                                           n_folds_tune,
                                                           par_grid['ml_g'], par_grid['ml_m'])
        g0_params = g0_params * n_folds
        m_params = m_params * n_folds
        g1_params = g1_params * n_folds

    res_manual = fit_apo(y, x, d, clone(learner_g), clone(learner_m),
                         treatment_level,
                         all_smpls, score,
                         normalize_ipw=normalize_ipw,
                         g0_params=g0_params, g1_params=g1_params, m_params=m_params)

    res_dict = {'coef': dml_obj.coef,
                'coef_manual': res_manual['theta'],
                'se': dml_obj.se,
                'se_manual': res_manual['se'],
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_t_stat = boot_apo(y, d, treatment_level, res_manual['thetas'], res_manual['ses'],
                               res_manual['all_g_hat0'], res_manual['all_g_hat1'],
                               res_manual['all_m_hat'],
                               all_smpls, score, bootstrap, n_rep_boot,
                               normalize_ipw=normalize_ipw)

        np.random.seed(3141)
        dml_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_t_stat' + bootstrap] = dml_obj.boot_t_stat
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat.reshape(-1, 1, 1)

    return res_dict


@pytest.mark.ci
def test_dml_apo_tune_coef(dml_apo_tune_fixture):
    assert math.isclose(dml_apo_tune_fixture['coef'],
                        dml_apo_tune_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_apo_tune_se(dml_apo_tune_fixture):
    assert math.isclose(dml_apo_tune_fixture['se'],
                        dml_apo_tune_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_apo_tune_boot(dml_apo_tune_fixture):
    for bootstrap in dml_apo_tune_fixture['boot_methods']:
        assert np.allclose(dml_apo_tune_fixture['boot_t_stat' + bootstrap],
                           dml_apo_tune_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
