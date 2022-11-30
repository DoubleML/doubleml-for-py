import numpy as np
import pytest

import doubleml as dml
from doubleml.datasets import make_pliv_multiway_cluster_CKMS2021

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from ._utils import draw_smpls
from ._utils_qte_manual import fit_qte, boot_qte

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


@pytest.fixture(scope="module")
def dml_qte_fixture(generate_data_quantiles, learner, dml_procedure):
    n_folds = 3
    n_rep = 1
    boot_methods = ['normal']
    n_rep_boot = 499

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
                                  trimming_threshold=1e-12)

    dml_qte_obj.fit()

    np.random.seed(42)
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds, n_rep=1)
    res_manual = fit_qte(y, x, d, quantiles, ml_g, ml_g, all_smpls,
                         n_rep=n_rep, dml_procedure=dml_procedure,
                         trimming_rule='truncate', trimming_threshold=1e-12, h=None,
                         normalize=True, draw_sample_splitting=True)

    res_dict = {'coef': dml_qte_obj.coef,
                'coef_manual': res_manual['qte'],
                'se': dml_qte_obj.se,
                'se_manual': res_manual['se'],
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(42)
        boot_qte_coef, boot_t_stat = boot_qte(res_manual['scaled_scores'], res_manual['ses'], quantiles,
                                              all_smpls, n_rep, bootstrap, n_rep_boot, apply_cross_fitting=True)
        np.random.seed(42)
        dml_qte_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_qte' + bootstrap] = dml_qte_obj.boot_coef
        res_dict['boot_t_stat' + bootstrap] = dml_qte_obj.boot_t_stat
        res_dict['boot_qte' + bootstrap + '_manual'] = boot_qte_coef
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat

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
def test_doubleml_qte_exceptions():
    np.random.seed(42)
    (x, y, d) = make_irm_data(100, 5, 2, return_type='array')
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d)
    ml_g = RandomForestClassifier()
    ml_m = RandomForestClassifier()

    msg = r'Quantiles have be between 0 or 1. Quantiles \[0.2 2. \] passed.'
    with pytest.raises(ValueError, match=msg):
        _ = dml.DoubleMLQTE(obj_dml_data, ml_g, ml_m, quantiles=[0.2, 2])


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
