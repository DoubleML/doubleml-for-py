import numpy as np
import pandas as pd
import pytest
import math

from sklearn.base import clone

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import doubleml as dml
from doubleml.datasets import make_irm_data

from ._utils import draw_smpls
from ._utils_irm_manual import fit_irm, boot_irm


@pytest.fixture(scope='module',
                params=[[LinearRegression(),
                         LogisticRegression(solver='lbfgs', max_iter=250)],
                        [RandomForestRegressor(max_depth=5, n_estimators=10),
                         RandomForestClassifier(max_depth=5, n_estimators=10)]])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['ATE', 'ATTE'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope='module',
                params=[True, False])
def normalize_ipw(request):
    return request.param


@pytest.fixture(scope='module',
                params=[0.01, 0.05])
def trimming_threshold(request):
    return request.param


@pytest.fixture(scope='module')
def dml_irm_fixture(generate_data_irm, learner, score, dml_procedure, normalize_ipw, trimming_threshold):
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
                                  dml_procedure=dml_procedure,
                                  normalize_ipw=normalize_ipw,
                                  draw_sample_splitting=False,
                                  trimming_threshold=trimming_threshold)

    # synchronize the sample splitting
    dml_irm_obj.set_sample_splitting(all_smpls=all_smpls)
    dml_irm_obj.fit()

    np.random.seed(3141)
    res_manual = fit_irm(y, x, d,
                         clone(learner[0]), clone(learner[1]),
                         all_smpls, dml_procedure, score,
                         normalize_ipw=normalize_ipw,
                         trimming_threshold=trimming_threshold)

    res_dict = {'coef': dml_irm_obj.coef,
                'coef_manual': res_manual['theta'],
                'se': dml_irm_obj.se,
                'se_manual': res_manual['se'],
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_theta, boot_t_stat = boot_irm(y, d, res_manual['thetas'], res_manual['ses'],
                                           res_manual['all_g_hat0'], res_manual['all_g_hat1'],
                                           res_manual['all_m_hat'], res_manual['all_p_hat'],
                                           all_smpls, score, bootstrap, n_rep_boot,
                                           dml_procedure=dml_procedure,
                                           normalize_ipw=normalize_ipw)

        np.random.seed(3141)
        dml_irm_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_coef' + bootstrap] = dml_irm_obj.boot_coef
        res_dict['boot_t_stat' + bootstrap] = dml_irm_obj.boot_t_stat
        res_dict['boot_coef' + bootstrap + '_manual'] = boot_theta
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat

    return res_dict


@pytest.mark.ci
def test_dml_irm_coef(dml_irm_fixture):
    assert math.isclose(dml_irm_fixture['coef'],
                        dml_irm_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_irm_se(dml_irm_fixture):
    assert math.isclose(dml_irm_fixture['se'],
                        dml_irm_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_irm_boot(dml_irm_fixture):
    for bootstrap in dml_irm_fixture['boot_methods']:
        assert np.allclose(dml_irm_fixture['boot_coef' + bootstrap],
                           dml_irm_fixture['boot_coef' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_irm_fixture['boot_t_stat' + bootstrap],
                           dml_irm_fixture['boot_t_stat' + bootstrap + '_manual'],
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
    assert isinstance(cate, dml.double_ml_blp.DoubleMLBLP)
    assert isinstance(cate.confint(), pd.DataFrame)

    groups_1 = pd.DataFrame(np.column_stack([obj_dml_data.data['X1'] <= 0,
                                             obj_dml_data.data['X1'] > 0.2]),
                            columns=['Group 1', 'Group 2'])
    msg = ('At least one group effect is estimated with less than 6 observations.')
    with pytest.warns(UserWarning, match=msg):
        gate_1 = dml_irm_obj.gate(groups_1)
    assert isinstance(gate_1, dml.double_ml_blp.DoubleMLBLP)
    assert isinstance(gate_1.confint(), pd.DataFrame)
    assert all(gate_1.confint().index == groups_1.columns)

    np.random.seed(42)
    groups_2 = pd.DataFrame(np.random.choice(["1", "2"], n))
    msg = ('At least one group effect is estimated with less than 6 observations.')
    with pytest.warns(UserWarning, match=msg):
        gate_2 = dml_irm_obj.gate(groups_2)
    assert isinstance(gate_2, dml.double_ml_blp.DoubleMLBLP)
    assert isinstance(gate_2.confint(), pd.DataFrame)
    assert all(gate_2.confint().index == ["Group_1", "Group_2"])
