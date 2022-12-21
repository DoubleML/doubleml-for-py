import numpy as np
import pandas as pd
import pytest

import doubleml as dml
from doubleml.datasets import make_pliv_multiway_cluster_CKMS2021

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from ._utils import draw_smpls
from ._utils_qte_manual import fit_qte, boot_qte, confint_qte

from doubleml.datasets import make_irm_data


@pytest.fixture(scope='module',
                params=[RandomForestClassifier(max_depth=2, n_estimators=10, random_state=42),
                        LogisticRegression()])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope='module',
                params=[None, 0.2])
def bandwidth(request):
    return request.param


@pytest.fixture(scope="module")
def dml_qte_fixture(generate_data_quantiles, learner, dml_procedure, bandwidth):
    n_folds = 3
    n_rep = 1
    boot_methods = ['normal']
    n_rep_boot = 2

    # collect data
    (x, y, d) = generate_data_quantiles
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d)

    # Set machine learning methods for g & m
    ml_g = clone(learner)
    ml_m = clone(learner)

    np.random.seed(42)
    quantiles = [0.25, 0.5, 0.75]
    dml_qte_obj = dml.DoubleMLQTE(obj_dml_data,
                                  ml_g, ml_m,
                                  quantiles=quantiles,
                                  n_folds=n_folds,
                                  n_rep=n_rep,
                                  dml_procedure=dml_procedure,
                                  trimming_threshold=1e-12,
                                  h=bandwidth)

    dml_qte_obj.fit()

    np.random.seed(42)
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds, n_rep=1, groups=d)
    res_manual = fit_qte(y, x, d, quantiles, ml_g, ml_g, all_smpls,
                         n_rep=n_rep, dml_procedure=dml_procedure,
                         trimming_rule='truncate', trimming_threshold=1e-12, h=bandwidth,
                         normalize=True, draw_sample_splitting=True)

    ci = dml_qte_obj.confint(joint=False, level=0.95)
    ci_manual = confint_qte(res_manual['qte'], res_manual['se'], quantiles,
                            boot_t_stat=None, joint=False, level=0.95)
    res_dict = {'coef': dml_qte_obj.coef,
                'coef_manual': res_manual['qte'],
                'se': dml_qte_obj.se,
                'se_manual': res_manual['se'],
                'boot_methods': boot_methods,
                'ci': ci.to_numpy(),
                'ci_manual': ci_manual.to_numpy(),
                'qte_model': dml_qte_obj}

    for bootstrap in boot_methods:
        np.random.seed(42)
        boot_qte_coef, boot_t_stat = boot_qte(res_manual['scaled_scores'], res_manual['ses'], quantiles,
                                              all_smpls, n_rep, bootstrap, n_rep_boot, apply_cross_fitting=True)

        np.random.seed(42)
        dml_qte_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)

        res_dict['boot_qte_' + bootstrap] = dml_qte_obj.boot_coef
        res_dict['boot_t_stat_' + bootstrap] = dml_qte_obj.boot_t_stat
        res_dict['boot_qte_' + bootstrap + '_manual'] = boot_qte_coef
        res_dict['boot_t_stat_' + bootstrap + '_manual'] = boot_t_stat

        ci = dml_qte_obj.confint(joint=True, level=0.95)
        ci_manual = confint_qte(res_manual['qte'], res_manual['se'], quantiles,
                                boot_t_stat=boot_t_stat, joint=True, level=0.95)
        res_dict['boot_ci_' + bootstrap] = ci.to_numpy()
        res_dict['boot_ci_' + bootstrap + '_manual'] = ci_manual.to_numpy()
    return res_dict


@pytest.mark.ci
def test_dml_qte_coef(dml_qte_fixture):
    assert np.allclose(dml_qte_fixture['coef'],
                       dml_qte_fixture['coef_manual'],
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_qte_se(dml_qte_fixture):
    assert np.allclose(dml_qte_fixture['se'],
                       dml_qte_fixture['se_manual'],
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_qte_boot(dml_qte_fixture):
    for bootstrap in dml_qte_fixture['boot_methods']:
        assert np.allclose(dml_qte_fixture['boot_qte_' + bootstrap],
                           dml_qte_fixture['boot_qte_' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_qte_fixture['boot_t_stat_' + bootstrap],
                           dml_qte_fixture['boot_t_stat_' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_qte_ci(dml_qte_fixture):
    assert np.allclose(dml_qte_fixture['ci'],
                       dml_qte_fixture['ci_manual'],
                       rtol=1e-9, atol=1e-4)
    for bootstrap in dml_qte_fixture['boot_methods']:
        assert np.allclose(dml_qte_fixture['boot_ci_' + bootstrap],
                           dml_qte_fixture['boot_ci_' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_doubleml_qte_exceptions():
    np.random.seed(42)
    (x, y, d) = make_irm_data(1000, 5, 2, return_type='array')
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d)
    ml_g = RandomForestClassifier(n_estimators=20)
    ml_m = RandomForestClassifier(n_estimators=20)

    msg = r'Quantiles have be between 0 or 1. Quantiles \[0.2 2. \] passed.'
    with pytest.raises(ValueError, match=msg):
        _ = dml.DoubleMLQTE(obj_dml_data, ml_g, ml_m, quantiles=[0.2, 2])

    msg = 'Invalid score pq. Valid scores PQ or LPQ.'
    with pytest.raises(ValueError, match=msg):
        _ = dml.DoubleMLQTE(obj_dml_data, ml_g, ml_m, score='pq')

    msg = 'Invalid trimming_rule discard. Valid trimming_rule truncate.'
    with pytest.raises(ValueError, match=msg):
        _ = dml.DoubleMLQTE(obj_dml_data, ml_g, ml_m, trimming_rule='discard')

    msg = r'Sample splitting not specified. Draw samples via .draw_sample splitting\(\). ' \
          'External samples not implemented yet.'
    with pytest.raises(ValueError, match=msg):
        dml_obj = dml.DoubleMLQTE(obj_dml_data, ml_g, ml_m, draw_sample_splitting=False)
        _ = dml_obj.smpls

    # bootstrap and ci
    dml_obj = dml.DoubleMLQTE(obj_dml_data, ml_g, ml_m)
    msg = r'Apply fit\(\) before bootstrap\(\).'
    with pytest.raises(ValueError, match=msg):
        dml_obj.bootstrap()

    dml_obj.fit()
    msg = 'Method must be "Bayes", "normal" or "wild". Got Normal.'
    with pytest.raises(ValueError, match=msg):
        dml_obj.bootstrap(method="Normal")
    msg = "The number of bootstrap replications must be of int type. Object of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_obj.bootstrap(n_rep_boot="100")
    msg = "The number of bootstrap replications must be positive. 0 was passed."
    with pytest.raises(ValueError, match=msg):
        dml_obj.bootstrap(n_rep_boot=0)


@pytest.mark.ci
def test_doubleml_cluster_not_implemented_exception():
    np.random.seed(3141)
    (x, y, d, cluster_vars, _) = make_pliv_multiway_cluster_CKMS2021(return_type='array')
    d = np.zeros_like(d)
    dml_data = dml.DoubleMLClusterData.from_arrays(x, y, d, cluster_vars)
    ml_g = RandomForestClassifier()
    ml_m = RandomForestClassifier()
    msg = 'Estimation with clustering not implemented.'
    with pytest.raises(NotImplementedError, match=msg):
        _ = dml.DoubleMLQTE(dml_data, ml_g, ml_m)


def test_doubleml_qte_return_types(dml_qte_fixture):
    assert isinstance(dml_qte_fixture['qte_model'].__str__(), str)
    assert isinstance(dml_qte_fixture['qte_model'].summary, pd.DataFrame)
