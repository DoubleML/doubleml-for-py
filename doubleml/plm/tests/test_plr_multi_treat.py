import numpy as np
import pytest

from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

import doubleml as dml

from ...tests._utils import draw_smpls, _clone
from ._utils_plr_manual import fit_plr_multitreat, boot_plr_multitreat, fit_sensitivity_elements_plr


@pytest.fixture(scope='module',
                params=range(2))
def idx(request):
    return request.param


@pytest.fixture(scope='module',
                params=[Lasso(alpha=0.1),
                        RandomForestRegressor(max_depth=2, n_estimators=10)])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['IV-type', 'partialling out'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope='module')
def dml_plr_multitreat_fixture(generate_data_bivariate, generate_data_toeplitz, idx, learner,
                               score, n_rep):
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
    n_coefs = len(d_cols)

    # Set machine learning methods for l, m & g
    ml_l = _clone(learner)
    ml_m = _clone(learner)
    if score == 'IV-type':
        ml_g = _clone(learner)
    else:
        ml_g = None

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData(data, 'y', d_cols, x_cols)
    dml_plr_obj = dml.DoubleMLPLR(obj_dml_data,
                                  ml_l, ml_m, ml_g,
                                  n_folds, n_rep,
                                  score=score)

    dml_plr_obj.fit()

    np.random.seed(3141)
    y = data['y'].values
    x = data.loc[:, x_cols].values
    d = data.loc[:, d_cols].values
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds, n_rep)

    res_manual = fit_plr_multitreat(y, x, d,
                                    _clone(learner), _clone(learner), _clone(learner),
                                    all_smpls, score, n_rep=n_rep)

    res_dict = {'coef': dml_plr_obj.coef,
                'coef_manual': res_manual['theta'],
                'se': dml_plr_obj.se,
                'se_manual': res_manual['se'],
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_t_stat = boot_plr_multitreat(
            y, d,
            res_manual['thetas'], res_manual['ses'],
            res_manual['all_l_hat'], res_manual['all_m_hat'], res_manual['all_g_hat'],
            all_smpls, score,
            bootstrap, n_rep_boot, n_rep)

        np.random.seed(3141)
        dml_plr_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_t_stat' + bootstrap] = dml_plr_obj.boot_t_stat
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat.reshape(-1, n_coefs, n_rep)

    # sensitivity tests
    res_dict['sensitivity_elements'] = dml_plr_obj.sensitivity_elements
    res_dict['sensitivity_elements_manual'] = fit_sensitivity_elements_plr(y, d,
                                                                           all_coef=dml_plr_obj.all_coef,
                                                                           predictions=dml_plr_obj.predictions,
                                                                           score=score,
                                                                           n_rep=n_rep)
    # check if sensitivity score with rho=0 gives equal asymptotic standard deviation
    dml_plr_obj.sensitivity_analysis(rho=0.0)
    res_dict['sensitivity_ses'] = dml_plr_obj.sensitivity_params['se']
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
        assert np.allclose(dml_plr_multitreat_fixture['boot_t_stat' + bootstrap],
                           dml_plr_multitreat_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_plr_multitreat_sensitivity(dml_plr_multitreat_fixture):
    sensitivity_element_names = ['sigma2', 'nu2', 'psi_sigma2', 'psi_nu2']
    for sensitivity_element in sensitivity_element_names:
        assert np.allclose(dml_plr_multitreat_fixture['sensitivity_elements'][sensitivity_element],
                           dml_plr_multitreat_fixture['sensitivity_elements_manual'][sensitivity_element])


@pytest.mark.ci
def test_dml_plr_multitreat_sensitivity_rho0(dml_plr_multitreat_fixture):
    assert np.allclose(dml_plr_multitreat_fixture['se'],
                       dml_plr_multitreat_fixture['sensitivity_ses']['lower'],
                       rtol=1e-9, atol=1e-4)
    assert np.allclose(dml_plr_multitreat_fixture['se'],
                       dml_plr_multitreat_fixture['sensitivity_ses']['upper'],
                       rtol=1e-9, atol=1e-4)
