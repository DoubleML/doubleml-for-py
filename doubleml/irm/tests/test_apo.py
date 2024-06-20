import numpy as np
import pandas as pd
import pytest
import math

from sklearn.base import clone

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import doubleml as dml
from doubleml.datasets import make_irm_data_discrete_treatements
from doubleml.utils.resampling import DoubleMLResampling

from ...tests._utils import draw_smpls
from ._utils_apo_manual import fit_apo, boot_apo, fit_sensitivity_elements_apo


@pytest.fixture(scope='module',
                params=[[LinearRegression(),
                         LogisticRegression(solver='lbfgs', max_iter=250, random_state=42)],
                        [RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
                         RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42)]])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=[False, True])
def normalize_ipw(request):
    return request.param


@pytest.fixture(scope='module',
                params=[0.2, 0.15])
def trimming_threshold(request):
    return request.param


@pytest.fixture(scope='module')
def dml_apo_fixture(generate_data_irm, learner, normalize_ipw, trimming_threshold):
    boot_methods = ['normal']
    n_folds = 2
    n_rep_boot = 499
    treatment_level = 0

    # Set machine learning methods for m & g
    ml_g = clone(learner[0])
    ml_m = clone(learner[1])

    np.random.seed(3141)
    n_obs = 100
    data_apo = make_irm_data_discrete_treatements(n_obs=n_obs)
    y = data_apo['y']
    x = data_apo['x']
    d = data_apo['d']
    df_apo = pd.DataFrame(
        np.column_stack((y, d, x)),
        columns=['y', 'd'] + ['x' + str(i) for i in range(data_apo['x'].shape[1])]
    )

    dml_data = dml.DoubleMLData(df_apo, 'y', 'd')
    all_smpls = draw_smpls(n_obs, n_folds, n_rep=1, groups=d)

    np.random.seed(3141)
    dml_obj = dml.DoubleMLAPO(dml_data,
                              ml_g, ml_m,
                              treatment_level=treatment_level,
                              n_folds=n_folds,
                              score='APO',
                              normalize_ipw=normalize_ipw,
                              draw_sample_splitting=False,
                              trimming_threshold=trimming_threshold)

    # synchronize the sample splitting
    dml_obj.set_sample_splitting(all_smpls=all_smpls)
    dml_obj.fit()

    np.random.seed(3141)
    res_manual = fit_apo(y, x, d,
                         clone(learner[0]), clone(learner[1]),
                         treatment_level=treatment_level,
                         all_smpls=all_smpls,
                         score='APO',
                         normalize_ipw=normalize_ipw,
                         trimming_threshold=trimming_threshold)

    np.random.seed(3141)
    # test with external nuisance predictions
    dml_obj_ext = dml.DoubleMLAPO(dml_data,
                                  ml_g, ml_m,
                                  treatment_level=treatment_level,
                                  n_folds=n_folds,
                                  score='APO',
                                  normalize_ipw=normalize_ipw,
                                  draw_sample_splitting=False,
                                  trimming_threshold=trimming_threshold)

    # synchronize the sample splitting
    dml_obj_ext.set_sample_splitting(all_smpls=all_smpls)

    prediction_dict = {'d': {'ml_g0': dml_obj.predictions['ml_g0'].reshape(-1, 1),
                             'ml_g1': dml_obj.predictions['ml_g1'].reshape(-1, 1),
                             'ml_m': dml_obj.predictions['ml_m'].reshape(-1, 1)}}
    dml_obj_ext.fit(external_predictions=prediction_dict)

    res_dict = {'coef': dml_obj.coef,
                'coef_manual': res_manual['theta'],
                'coef_ext': dml_obj_ext.coef,
                'se': dml_obj.se,
                'se_manual': res_manual['se'],
                'se_ext': dml_obj_ext.se,
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_t_stat = boot_apo(y, d, treatment_level, res_manual['thetas'], res_manual['ses'],
                               res_manual['all_g_hat0'], res_manual['all_g_hat1'],
                               res_manual['all_m_hat'],
                               all_smpls,
                               score='APO',
                               bootstrap=bootstrap,
                               n_rep_boot=n_rep_boot,
                               normalize_ipw=normalize_ipw)

        np.random.seed(3141)
        dml_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        np.random.seed(3141)
        dml_obj_ext.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_t_stat' + bootstrap] = dml_obj.boot_t_stat
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat.reshape(-1, 1, 1)
        res_dict['boot_t_stat' + bootstrap + '_ext'] = dml_obj_ext.boot_t_stat

    # check if sensitivity score with rho=0 gives equal asymptotic standard deviation
    dml_obj.sensitivity_analysis(rho=0.0)
    res_dict['sensitivity_ses'] = dml_obj.sensitivity_params['se']

    # sensitivity tests
    res_dict['sensitivity_elements'] = dml_obj.sensitivity_elements
    res_dict['sensitivity_elements_manual'] = fit_sensitivity_elements_apo(y, d,
                                                                           treatment_level,
                                                                           all_coef=dml_obj.all_coef,
                                                                           predictions=dml_obj.predictions,
                                                                           score='APO',
                                                                           n_rep=1)
    return res_dict


@pytest.mark.ci
def test_dml_apo_coef(dml_apo_fixture):
    assert math.isclose(dml_apo_fixture['coef'][0],
                        dml_apo_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)
    assert math.isclose(dml_apo_fixture['coef'][0],
                        dml_apo_fixture['coef_ext'][0],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_apo_se(dml_apo_fixture):
    assert math.isclose(dml_apo_fixture['se'][0],
                        dml_apo_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)
    assert math.isclose(dml_apo_fixture['se'][0],
                        dml_apo_fixture['se_ext'][0],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_apo_boot(dml_apo_fixture):
    for bootstrap in dml_apo_fixture['boot_methods']:
        assert np.allclose(dml_apo_fixture['boot_t_stat' + bootstrap],
                           dml_apo_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_apo_fixture['boot_t_stat' + bootstrap],
                           dml_apo_fixture['boot_t_stat' + bootstrap + '_ext'],
                           rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_apo_sensitivity_rho0(dml_apo_fixture):
    assert np.allclose(dml_apo_fixture['se'],
                       dml_apo_fixture['sensitivity_ses']['lower'],
                       rtol=1e-9, atol=1e-4)
    assert np.allclose(dml_apo_fixture['se'],
                       dml_apo_fixture['sensitivity_ses']['upper'],
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_apo_sensitivity(dml_apo_fixture):
    sensitivity_element_names = ['sigma2', 'nu2', 'psi_sigma2', 'psi_nu2']
    for sensitivity_element in sensitivity_element_names:
        assert np.allclose(dml_apo_fixture['sensitivity_elements'][sensitivity_element],
                           dml_apo_fixture['sensitivity_elements_manual'][sensitivity_element],
                           rtol=1e-9, atol=1e-4)
