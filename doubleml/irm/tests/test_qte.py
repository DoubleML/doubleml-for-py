import numpy as np
import pandas as pd
import pytest
import copy

import doubleml as dml

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from ...tests._utils import draw_smpls
from ._utils_qte_manual import fit_qte, boot_qte, confint_qte

from doubleml.datasets import make_irm_data
from ...utils._estimation import _default_kde


quantiles = [0.25, 0.5, 0.75]
n_quantiles = len(quantiles)
n_rep = 1


@pytest.fixture(scope='module',
                params=[RandomForestClassifier(max_depth=2, n_estimators=10, random_state=42),
                        LogisticRegression()])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=[True, False])
def normalize_ipw(request):
    return request.param


@pytest.fixture(scope='module',
                params=[None, _default_kde])
def kde(request):
    return request.param


@pytest.fixture(scope="module")
def dml_qte_fixture(generate_data_quantiles, learner, normalize_ipw, kde):
    n_folds = 3
    boot_methods = ['normal']
    n_rep_boot = 2

    # collect data
    (x, y, d) = generate_data_quantiles
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d)

    # Set machine learning methods for g & m
    ml_g = clone(learner)
    ml_m = clone(learner)

    np.random.seed(42)
    dml_qte_obj = dml.DoubleMLQTE(obj_dml_data,
                                  ml_g, ml_m,
                                  quantiles=quantiles,
                                  n_folds=n_folds,
                                  n_rep=n_rep,
                                  normalize_ipw=normalize_ipw,
                                  trimming_threshold=1e-12,
                                  kde=kde)
    unfitted_qte_model = copy.copy(dml_qte_obj)
    dml_qte_obj.fit()

    np.random.seed(42)
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds, n_rep=1, groups=d)
    res_manual = fit_qte(y, x, d, quantiles, ml_g, ml_g, all_smpls,
                         n_rep=n_rep,
                         normalize_ipw=normalize_ipw,
                         trimming_rule='truncate', trimming_threshold=1e-12, kde=kde,
                         draw_sample_splitting=True)

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
                'qte_model': dml_qte_obj,
                'unfitted_qte_model': unfitted_qte_model}

    for bootstrap in boot_methods:
        np.random.seed(42)
        boot_t_stat = boot_qte(res_manual['scaled_scores'], res_manual['ses'], quantiles,
                               all_smpls, n_rep, bootstrap, n_rep_boot)

        np.random.seed(42)
        dml_qte_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)

        res_dict['boot_t_stat_' + bootstrap] = dml_qte_obj.boot_t_stat
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

    msg = r'Sample splitting not specified. Draw samples via .draw_sample splitting\(\). ' \
          'External samples not implemented yet.'
    with pytest.raises(ValueError, match=msg):
        dml_obj = dml.DoubleMLQTE(obj_dml_data, ml_g, ml_m, draw_sample_splitting=False)
        _ = dml_obj.smpls


def test_doubleml_qte_return_types(dml_qte_fixture):
    assert isinstance(dml_qte_fixture['qte_model'].__str__(), str)
    assert isinstance(dml_qte_fixture['qte_model'].summary, pd.DataFrame)

    assert dml_qte_fixture['qte_model'].all_coef.shape == (n_quantiles, n_rep)
    assert isinstance(dml_qte_fixture['unfitted_qte_model'].summary, pd.DataFrame)
