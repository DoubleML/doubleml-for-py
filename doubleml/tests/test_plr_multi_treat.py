import numpy as np
import pytest

from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.linear_model import Lasso

import doubleml as dml

from ._utils_plr_manual import plr_dml1, plr_dml2, fit_nuisance_plr, boot_plr


@pytest.fixture(scope='module',
                params=range(2))
def idx(request):
    return request.param


@pytest.fixture(scope='module',
                params=[Lasso(alpha=0.1)])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['IV-type', 'partialling out'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope='module')
def dml_plr_multitreat_fixture(generate_data_bivariate, generate_data_toeplitz, idx, learner, score, dml_procedure):
    boot_methods = ['normal']
    n_folds = 2
    n_rep_boot = 483

    # collect data
    if idx == 0:
        data = generate_data_bivariate
    else:
        assert idx == 1
        data = generate_data_toeplitz
    x_cols = data.columns[data.columns.str.startswith('X')].tolist()
    d_cols = data.columns[data.columns.str.startswith('d')].tolist()

    # Set machine learning methods for m & g
    ml_g = clone(learner)
    ml_m = clone(learner)

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData(data, 'y', d_cols, x_cols)
    dml_plr_obj = dml.DoubleMLPLR(obj_dml_data,
                                  ml_g, ml_m,
                                  n_folds,
                                  score=score,
                                  dml_procedure=dml_procedure)

    dml_plr_obj.fit()

    np.random.seed(3141)
    y = data['y'].values
    x = data.loc[:, x_cols].values
    d = data.loc[:, d_cols].values
    resampling = KFold(n_splits=n_folds,
                       shuffle=True)
    smpls = [(train, test) for train, test in resampling.split(x)]

    n_d = d.shape[1]

    coef_manual = np.full(n_d, np.nan)
    se_manual = np.full(n_d, np.nan)

    all_g_hat = []
    all_m_hat = []

    for i_d in range(n_d):

        Xd = np.hstack((x, np.delete(d, i_d, axis=1)))

        g_hat, m_hat = fit_nuisance_plr(y, Xd, d[:, i_d],
                                        clone(learner), clone(learner), smpls)

        all_g_hat.append(g_hat)
        all_m_hat.append(m_hat)

        if dml_procedure == 'dml1':
            coef_manual[i_d], se_manual[i_d] = plr_dml1(y, Xd, d[:, i_d],
                                                        g_hat, m_hat,
                                                        smpls, score)
        else:
            assert dml_procedure == 'dml2'
            coef_manual[i_d], se_manual[i_d] = plr_dml2(y, Xd, d[:, i_d],
                                                        g_hat, m_hat,
                                                        smpls, score)

    res_dict = {'coef': dml_plr_obj.coef,
                'coef_manual': coef_manual,
                'se': dml_plr_obj.se,
                'se_manual': se_manual,
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_theta, boot_t_stat = boot_plr(coef_manual,
                                           y, d,
                                           all_g_hat, all_m_hat,
                                           smpls, score,
                                           se_manual,
                                           bootstrap, n_rep_boot,
                                           dml_procedure)

        np.random.seed(3141)
        dml_plr_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_coef' + bootstrap] = dml_plr_obj.boot_coef
        res_dict['boot_t_stat' + bootstrap] = dml_plr_obj.boot_t_stat
        res_dict['boot_coef' + bootstrap + '_manual'] = boot_theta
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat

    return res_dict


@pytest.mark.ci
def test_dml_plr_multitreat_coef(dml_plr_multitreat_fixture):
    assert np.allclose(dml_plr_multitreat_fixture['coef'],
                       dml_plr_multitreat_fixture['coef_manual'],
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_plr_multitreat_se(dml_plr_multitreat_fixture):
    assert np.allclose(dml_plr_multitreat_fixture['se'],
                       dml_plr_multitreat_fixture['se_manual'],
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_plr_multitreat_boot(dml_plr_multitreat_fixture):
    for bootstrap in dml_plr_multitreat_fixture['boot_methods']:
        assert np.allclose(dml_plr_multitreat_fixture['boot_coef' + bootstrap],
                           dml_plr_multitreat_fixture['boot_coef' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_plr_multitreat_fixture['boot_t_stat' + bootstrap],
                           dml_plr_multitreat_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
