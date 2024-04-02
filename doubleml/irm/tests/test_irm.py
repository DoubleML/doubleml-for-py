import numpy as np
import pandas as pd
import pytest
import math

from sklearn.base import clone

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import doubleml as dml
from doubleml.datasets import make_irm_data
from doubleml.utils.resampling import DoubleMLResampling

from ...tests._utils import draw_smpls
from ._utils_irm_manual import fit_irm, boot_irm, fit_sensitivity_elements_irm


@pytest.fixture(scope='module',
                params=[[LinearRegression(),
                         LogisticRegression(solver='lbfgs', max_iter=250)],
                        [RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
                         RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42)]])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['ATE', 'ATTE'])
def score(request):
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
def dml_irm_fixture(generate_data_irm, learner, score, normalize_ipw, trimming_threshold):
    boot_methods = ['normal']
    n_folds = 2
    n_rep_boot = 499

    # collect data
    (x, y, d) = generate_data_irm

    # Set machine learning methods for m & g
    ml_g = clone(learner[0])
    ml_m = clone(learner[1])

    np.random.seed(3141)
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds, n_rep=1, groups=d)
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d)

    np.random.seed(3141)
    dml_irm_obj = dml.DoubleMLIRM(obj_dml_data,
                                  ml_g, ml_m,
                                  n_folds,
                                  score=score,
                                  normalize_ipw=normalize_ipw,
                                  draw_sample_splitting=False,
                                  trimming_threshold=trimming_threshold)

    # synchronize the sample splitting
    dml_irm_obj.set_sample_splitting(all_smpls=all_smpls)
    dml_irm_obj.fit()

    np.random.seed(3141)
    res_manual = fit_irm(y, x, d,
                         clone(learner[0]), clone(learner[1]),
                         all_smpls, score,
                         normalize_ipw=normalize_ipw,
                         trimming_threshold=trimming_threshold)

    np.random.seed(3141)
    # test with external nuisance predictions
    dml_irm_obj_ext = dml.DoubleMLIRM(obj_dml_data,
                                      ml_g, ml_m,
                                      n_folds,
                                      score=score,
                                      normalize_ipw=normalize_ipw,
                                      draw_sample_splitting=False,
                                      trimming_threshold=trimming_threshold)

    # synchronize the sample splitting
    dml_irm_obj_ext.set_sample_splitting(all_smpls=all_smpls)

    prediction_dict = {'d': {'ml_g0': dml_irm_obj.predictions['ml_g0'].reshape(-1, 1),
                             'ml_g1': dml_irm_obj.predictions['ml_g1'].reshape(-1, 1),
                             'ml_m': dml_irm_obj.predictions['ml_m'].reshape(-1, 1)}}
    dml_irm_obj_ext.fit(external_predictions=prediction_dict)

    res_dict = {'coef': dml_irm_obj.coef,
                'coef_manual': res_manual['theta'],
                'coef_ext': dml_irm_obj_ext.coef,
                'se': dml_irm_obj.se,
                'se_manual': res_manual['se'],
                'se_ext': dml_irm_obj_ext.se,
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_t_stat = boot_irm(y, d, res_manual['thetas'], res_manual['ses'],
                               res_manual['all_g_hat0'], res_manual['all_g_hat1'],
                               res_manual['all_m_hat'], res_manual['all_p_hat'],
                               all_smpls, score, bootstrap, n_rep_boot,
                               normalize_ipw=normalize_ipw)

        np.random.seed(3141)
        dml_irm_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        np.random.seed(3141)
        dml_irm_obj_ext.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_t_stat' + bootstrap] = dml_irm_obj.boot_t_stat
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat.reshape(-1, 1, 1)
        res_dict['boot_t_stat' + bootstrap + '_ext'] = dml_irm_obj_ext.boot_t_stat

    # sensitivity tests
    res_dict['sensitivity_elements'] = dml_irm_obj.sensitivity_elements
    res_dict['sensitivity_elements_manual'] = fit_sensitivity_elements_irm(y, d,
                                                                           all_coef=dml_irm_obj.all_coef,
                                                                           predictions=dml_irm_obj.predictions,
                                                                           score=score,
                                                                           n_rep=1)

    # check if sensitivity score with rho=0 gives equal asymptotic standard deviation
    dml_irm_obj.sensitivity_analysis(rho=0.0)
    res_dict['sensitivity_ses'] = dml_irm_obj.sensitivity_params['se']
    return res_dict


@pytest.mark.ci
def test_dml_irm_coef(dml_irm_fixture):
    assert math.isclose(dml_irm_fixture['coef'][0],
                        dml_irm_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)
    assert math.isclose(dml_irm_fixture['coef'][0],
                        dml_irm_fixture['coef_ext'][0],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_irm_se(dml_irm_fixture):
    assert math.isclose(dml_irm_fixture['se'][0],
                        dml_irm_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)
    assert math.isclose(dml_irm_fixture['se'][0],
                        dml_irm_fixture['se_ext'][0],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_irm_boot(dml_irm_fixture):
    for bootstrap in dml_irm_fixture['boot_methods']:
        assert np.allclose(dml_irm_fixture['boot_t_stat' + bootstrap],
                           dml_irm_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_irm_fixture['boot_t_stat' + bootstrap],
                           dml_irm_fixture['boot_t_stat' + bootstrap + '_ext'],
                           rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_irm_sensitivity(dml_irm_fixture):
    sensitivity_element_names = ['sigma2', 'nu2', 'psi_sigma2', 'psi_nu2']
    for sensitivity_element in sensitivity_element_names:
        assert np.allclose(dml_irm_fixture['sensitivity_elements'][sensitivity_element],
                           dml_irm_fixture['sensitivity_elements_manual'][sensitivity_element],
                           rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_irm_sensitivity_rho0(dml_irm_fixture):
    assert np.allclose(dml_irm_fixture['se'],
                       dml_irm_fixture['sensitivity_ses']['lower'],
                       rtol=1e-9, atol=1e-4)
    assert np.allclose(dml_irm_fixture['se'],
                       dml_irm_fixture['sensitivity_ses']['upper'],
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_irm_cate_gate():
    n = 9
    # collect data
    np.random.seed(42)
    obj_dml_data = make_irm_data(n_obs=n, dim_x=2)

    # First stage estimation
    ml_g = RandomForestRegressor(n_estimators=10)
    ml_m = RandomForestClassifier(n_estimators=10)

    dml_irm_obj = dml.DoubleMLIRM(obj_dml_data,
                                  ml_m=ml_m,
                                  ml_g=ml_g,
                                  trimming_threshold=0.05,
                                  n_folds=5)

    dml_irm_obj.fit()
    # create a random basis
    random_basis = pd.DataFrame(np.random.normal(0, 1, size=(n, 5)))
    cate = dml_irm_obj.cate(random_basis)
    assert isinstance(cate, dml.utils.blp.DoubleMLBLP)
    assert isinstance(cate.confint(), pd.DataFrame)

    groups_1 = pd.DataFrame(np.column_stack([obj_dml_data.data['X1'] <= 0,
                                             obj_dml_data.data['X1'] > 0.2]),
                            columns=['Group 1', 'Group 2'])
    msg = ('At least one group effect is estimated with less than 6 observations.')
    with pytest.warns(UserWarning, match=msg):
        gate_1 = dml_irm_obj.gate(groups_1)
    assert isinstance(gate_1, dml.utils.blp.DoubleMLBLP)
    assert isinstance(gate_1.confint(), pd.DataFrame)
    assert all(gate_1.confint().index == groups_1.columns.to_list())

    np.random.seed(42)
    groups_2 = pd.DataFrame(np.random.choice(["1", "2"], n))
    msg = ('At least one group effect is estimated with less than 6 observations.')
    with pytest.warns(UserWarning, match=msg):
        gate_2 = dml_irm_obj.gate(groups_2)
    assert isinstance(gate_2, dml.utils.blp.DoubleMLBLP)
    assert isinstance(gate_2.confint(), pd.DataFrame)
    assert all(gate_2.confint().index == ["Group_1", "Group_2"])


@pytest.fixture(scope='module',
                params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope='module')
def dml_irm_weights_fixture(n_rep):
    n = 10000
    # collect data
    np.random.seed(42)
    obj_dml_data = make_irm_data(n_obs=n, dim_x=2)
    kwargs = {
        "trimming_threshold": 0.05,
        "n_folds": 5,
        "n_rep": n_rep,
        "draw_sample_splitting": False
    }

    smpls = DoubleMLResampling(
        n_folds=5,
        n_rep=n_rep,
        n_obs=n,
        stratify=obj_dml_data.d).split_samples()

    # First stage estimation
    ml_g = LinearRegression()
    ml_m = LogisticRegression(penalty='l2', random_state=42)

    # ATE with and without weights
    dml_irm_obj_ate_no_weights = dml.DoubleMLIRM(
        obj_dml_data,
        ml_g=clone(ml_g),
        ml_m=clone(ml_m),
        score='ATE',
        **kwargs)
    dml_irm_obj_ate_no_weights.set_sample_splitting(smpls)
    np.random.seed(42)
    dml_irm_obj_ate_no_weights.fit()

    dml_irm_obj_ate_weights = dml.DoubleMLIRM(
        obj_dml_data,
        ml_g=clone(ml_g),
        ml_m=clone(ml_m),
        score='ATE',
        weights=np.ones_like(obj_dml_data.y), **kwargs)
    dml_irm_obj_ate_weights.set_sample_splitting(smpls)
    np.random.seed(42)
    dml_irm_obj_ate_weights.fit()

    # ATTE with and without weights
    dml_irm_obj_atte_no_weights = dml.DoubleMLIRM(
        obj_dml_data,
        ml_g=clone(ml_g),
        ml_m=clone(ml_m),
        score='ATTE',
        **kwargs)
    dml_irm_obj_atte_no_weights.set_sample_splitting(smpls)
    np.random.seed(42)
    dml_irm_obj_atte_no_weights.fit()

    m_hat = dml_irm_obj_atte_no_weights.predictions["ml_m"][:, :, 0]
    p_hat = obj_dml_data.d.mean()
    weights = obj_dml_data.d / p_hat
    weights_bar = m_hat / p_hat
    weight_dict = {'weights': weights, 'weights_bar': weights_bar}
    dml_irm_obj_atte_weights = dml.DoubleMLIRM(
        obj_dml_data,
        ml_g=clone(ml_g),
        ml_m=clone(ml_m),
        score='ATE',
        weights=weight_dict, **kwargs)
    dml_irm_obj_atte_weights.set_sample_splitting(smpls)
    np.random.seed(42)
    dml_irm_obj_atte_weights.fit()

    res_dict = {
        'coef_ate': dml_irm_obj_ate_no_weights.coef,
        'coef_ate_weights': dml_irm_obj_ate_weights.coef,
        'coef_atte': dml_irm_obj_atte_no_weights.coef,
        'coef_atte_weights': dml_irm_obj_atte_weights.coef,
        'se_ate': dml_irm_obj_ate_no_weights.se,
        'se_ate_weights': dml_irm_obj_ate_weights.se,
        'se_atte': dml_irm_obj_atte_no_weights.se,
        'se_atte_weights': dml_irm_obj_atte_weights.se,
    }
    return res_dict


@pytest.mark.ci
def test_dml_irm_ate_weights(dml_irm_weights_fixture):
    assert math.isclose(dml_irm_weights_fixture['coef_ate'],
                        dml_irm_weights_fixture['coef_ate_weights'],
                        rel_tol=1e-9, abs_tol=1e-4)
    assert math.isclose(dml_irm_weights_fixture['se_ate'],
                        dml_irm_weights_fixture['se_ate_weights'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_irm_atte_weights(dml_irm_weights_fixture):
    assert math.isclose(dml_irm_weights_fixture['coef_atte'],
                        dml_irm_weights_fixture['coef_atte_weights'],
                        rel_tol=1e-9, abs_tol=1e-4)
    # Remark that the scores are slightly different (Y instead of g(1,X) and coefficient of theta)
    assert math.isclose(dml_irm_weights_fixture['se_atte'],
                        dml_irm_weights_fixture['se_atte_weights'],
                        rel_tol=1e-5, abs_tol=1e-3)
