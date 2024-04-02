import numpy as np
import pytest
import math

from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

import doubleml as dml

from ...tests._utils import draw_smpls
from ._utils_did_manual import fit_did, boot_did, tune_nuisance_did


@pytest.fixture(scope='module',
                params=[RandomForestRegressor(random_state=42)])
def learner_g(request):
    return request.param


@pytest.fixture(scope='module',
                params=[LogisticRegression()])
def learner_m(request):
    return request.param


@pytest.fixture(scope='module',
                params=['observational', 'experimental'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=[True, False])
def in_sample_normalization(request):
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
def dml_did_fixture(generate_data_did, learner_g, learner_m, score, in_sample_normalization,
                    tune_on_folds):
    par_grid = {'ml_g': get_par_grid(learner_g),
                'ml_m': get_par_grid(learner_m)}
    n_folds_tune = 4

    boot_methods = ['normal']
    n_folds = 2
    n_rep_boot = 499

    # collect data
    (x, y, d) = generate_data_did

    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds, n_rep=1, groups=d)
    # Set machine learning methods for m & g
    ml_g = clone(learner_g)
    ml_m = clone(learner_m)

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d)
    dml_did_obj = dml.DoubleMLDID(obj_dml_data,
                                  ml_g, ml_m,
                                  n_folds,
                                  score=score,
                                  in_sample_normalization=in_sample_normalization,
                                  draw_sample_splitting=False)
    # synchronize the sample splitting
    dml_did_obj.set_sample_splitting(all_smpls=all_smpls)

    # tune hyperparameters
    tune_res = dml_did_obj.tune(par_grid, tune_on_folds=tune_on_folds, n_folds_tune=n_folds_tune,
                                return_tune_res=False)
    assert isinstance(tune_res, dml.DoubleMLDID)

    dml_did_obj.fit()

    np.random.seed(3141)
    smpls = all_smpls[0]

    if tune_on_folds:
        g0_params, g1_params, m_params = tune_nuisance_did(y, x, d,
                                                           clone(learner_g), clone(learner_m), smpls, score,
                                                           n_folds_tune,
                                                           par_grid['ml_g'], par_grid['ml_m'])
    else:
        xx = [(np.arange(len(y)), np.array([]))]
        g0_params, g1_params, m_params = tune_nuisance_did(y, x, d,
                                                           clone(learner_g), clone(learner_m), xx, score,
                                                           n_folds_tune,
                                                           par_grid['ml_g'], par_grid['ml_m'])
        g0_params = g0_params * n_folds
        if score == 'experimental':
            g1_params = g1_params * n_folds
            m_params = None
        else:
            assert score == 'observational'
            g1_params = None
            m_params = m_params * n_folds

    res_manual = fit_did(y, x, d, clone(learner_g), clone(learner_m),
                         all_smpls, score, in_sample_normalization,
                         g0_params=g0_params, g1_params=g1_params, m_params=m_params)

    res_dict = {'coef': dml_did_obj.coef,
                'coef_manual': res_manual['theta'],
                'se': dml_did_obj.se,
                'se_manual': res_manual['se'],
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_t_stat = boot_did(y, res_manual['thetas'], res_manual['ses'],
                               res_manual['all_psi_a'], res_manual['all_psi_b'],
                               all_smpls, bootstrap, n_rep_boot)

        np.random.seed(3141)
        dml_did_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_t_stat' + bootstrap] = dml_did_obj.boot_t_stat
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat.reshape(-1, 1, 1)

    return res_dict


@pytest.mark.ci
def test_dml_did_coef(dml_did_fixture):
    assert math.isclose(dml_did_fixture['coef'][0],
                        dml_did_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_did_se(dml_did_fixture):
    assert math.isclose(dml_did_fixture['se'][0],
                        dml_did_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_did_boot(dml_did_fixture):
    for bootstrap in dml_did_fixture['boot_methods']:
        assert np.allclose(dml_did_fixture['boot_t_stat' + bootstrap],
                           dml_did_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
