import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import doubleml as dml
from doubleml.datasets import make_irm_data_discrete_treatments

from ._utils_apos_manual import fit_apos, boot_apos
from ...tests._utils import confint_manual


@pytest.fixture(scope='module',
                params=[[LogisticRegression(solver='lbfgs', max_iter=250),
                         LogisticRegression(solver='lbfgs', max_iter=250)],
                        [RandomForestClassifier(max_depth=2, n_estimators=10, random_state=42),
                         RandomForestClassifier(max_depth=2, n_estimators=10, random_state=42)]])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=[1])
def n_rep(request):
    return request.param


@pytest.fixture(scope='module',
                params=[False, True])
def normalize_ipw(request):
    return request.param


@pytest.fixture(scope='module',
                params=[0.2, 0.15])
def trimming_threshold(request):
    return request.param


@pytest.fixture(scope='module',
                params=[[0, 1, 2], [0]])
def treatment_levels(request):
    return request.param


@pytest.fixture(scope='module')
def dml_apos_classifier_fixture(learner, n_rep, normalize_ipw, trimming_threshold, treatment_levels):
    boot_methods = ['normal']
    n_folds = 2
    n_rep_boot = 499

    np.random.seed(3141)
    n_obs = 500
    data = make_irm_data_discrete_treatments(n_obs=n_obs)
    y = np.random.binomial(1, 0.5, n_obs)
    x = data['x']
    d = data['d']
    df = pd.DataFrame(
        np.column_stack((y, d, x)),
        columns=['y', 'd'] + ['x' + str(i) for i in range(data['x'].shape[1])]
    )

    dml_data = dml.DoubleMLData(df, 'y', 'd')

    input_args = {
        'obj_dml_data': dml_data,
        'ml_g': clone(learner[0]),
        'ml_m': clone(learner[1]),
        "treatment_levels": treatment_levels,
        "n_folds": n_folds,
        "n_rep": n_rep,
        "score": 'APO',
        "normalize_ipw": normalize_ipw,
        "trimming_rule": 'truncate',
        "trimming_threshold": trimming_threshold,
        }

    unfitted_apos_model = dml.DoubleMLAPOS(**input_args)
    np.random.seed(42)
    dml_obj = dml.DoubleMLAPOS(**input_args)
    dml_obj.fit()
    # get the sample splitting
    all_smpls = dml_obj.smpls

    np.random.seed(42)
    dml_obj_ext_smpls = dml.DoubleMLAPOS(**input_args, draw_sample_splitting=False)
    dml_obj_ext_smpls.set_sample_splitting(dml_obj.smpls)
    dml_obj_ext_smpls.fit()

    np.random.seed(42)
    res_manual = fit_apos(
        y, x, d,
        clone(learner[0]), clone(learner[1]),
        treatment_levels=treatment_levels,
        all_smpls=all_smpls,
        score='APO',
        trimming_rule='truncate',
        normalize_ipw=normalize_ipw,
        trimming_threshold=trimming_threshold)

    ci = dml_obj.confint(joint=False, level=0.95)
    ci_ext_smpls = dml_obj_ext_smpls.confint(joint=False, level=0.95)
    ci_manual = confint_manual(
        res_manual['apos'], res_manual['se'], treatment_levels,
        boot_t_stat=None, joint=False, level=0.95
        )

    res_dict = {
        'coef': dml_obj.coef,
        'coef_ext_smpls': dml_obj_ext_smpls.coef,
        'coef_manual': res_manual['apos'],
        'se': dml_obj.se,
        'se_ext_smpls': dml_obj_ext_smpls.se,
        'se_manual': res_manual['se'],
        'boot_methods': boot_methods,
        'n_treatment_levels': len(treatment_levels),
        'n_rep': n_rep,
        'ci': ci.to_numpy(),
        'ci_ext_smpls': ci_ext_smpls.to_numpy(),
        'ci_manual': ci_manual.to_numpy(),
        'apos_model': dml_obj,
        'unfitted_apos_model': unfitted_apos_model
    }

    for bootstrap in boot_methods:
        np.random.seed(42)
        boot_t_stat = boot_apos(res_manual['apo_scaled_score'], res_manual['all_se'], treatment_levels,
                                all_smpls, n_rep, bootstrap, n_rep_boot)

        np.random.seed(42)
        dml_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)

        res_dict['boot_t_stat_' + bootstrap] = dml_obj.boot_t_stat
        res_dict['boot_t_stat_' + bootstrap + '_manual'] = boot_t_stat

        ci = dml_obj.confint(joint=True, level=0.95)
        ci_manual = confint_manual(res_manual['apos'], res_manual['se'], treatment_levels,
                                   boot_t_stat=boot_t_stat, joint=True, level=0.95)
        res_dict['boot_ci_' + bootstrap] = ci.to_numpy()
        res_dict['boot_ci_' + bootstrap + '_manual'] = ci_manual.to_numpy()

    return res_dict


@pytest.mark.ci
def test_dml_apos_coef(dml_apos_classifier_fixture):
    assert np.allclose(dml_apos_classifier_fixture['coef'],
                       dml_apos_classifier_fixture['coef_manual'],
                       rtol=1e-9, atol=1e-9)
    assert np.allclose(dml_apos_classifier_fixture['coef'],
                       dml_apos_classifier_fixture['coef_ext_smpls'],
                       rtol=1e-9, atol=1e-9)


@pytest.mark.ci
def test_dml_apos_se(dml_apos_classifier_fixture):
    assert np.allclose(dml_apos_classifier_fixture['se'],
                       dml_apos_classifier_fixture['se_manual'],
                       rtol=1e-9, atol=1e-9)
    assert np.allclose(dml_apos_classifier_fixture['se'],
                       dml_apos_classifier_fixture['se_ext_smpls'],
                       rtol=1e-9, atol=1e-9)


@pytest.mark.ci
def test_dml_apos_boot(dml_apos_classifier_fixture):
    for bootstrap in dml_apos_classifier_fixture['boot_methods']:
        assert np.allclose(dml_apos_classifier_fixture['boot_t_stat_' + bootstrap],
                           dml_apos_classifier_fixture['boot_t_stat_' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_apos_ci(dml_apos_classifier_fixture):
    for bootstrap in dml_apos_classifier_fixture['boot_methods']:
        assert np.allclose(dml_apos_classifier_fixture['ci'],
                           dml_apos_classifier_fixture['ci_manual'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_apos_classifier_fixture['ci'],
                           dml_apos_classifier_fixture['ci_ext_smpls'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_apos_classifier_fixture['boot_ci_' + bootstrap],
                           dml_apos_classifier_fixture['boot_ci_' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_doubleml_apos_return_types(dml_apos_classifier_fixture):
    assert isinstance(dml_apos_classifier_fixture['apos_model'].__str__(), str)
    assert isinstance(dml_apos_classifier_fixture['apos_model'].summary, pd.DataFrame)

    assert dml_apos_classifier_fixture['apos_model'].all_coef.shape == (
        dml_apos_classifier_fixture['n_treatment_levels'],
        dml_apos_classifier_fixture['n_rep']
    )
    assert isinstance(dml_apos_classifier_fixture['unfitted_apos_model'].summary, pd.DataFrame)
