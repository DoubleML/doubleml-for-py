import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import doubleml as dml
from doubleml.datasets import make_irm_data_discrete_treatments, make_irm_data

from ._utils_apos_manual import fit_apos, boot_apos
from ...tests._utils import confint_manual


@pytest.mark.ci
def test_apo_properties():
    n = 20
    # collect data
    np.random.seed(42)
    obj_dml_data = make_irm_data(n_obs=n, dim_x=2)

    dml_obj = dml.DoubleMLAPOS(obj_dml_data,
                               ml_g=RandomForestRegressor(n_estimators=10),
                               ml_m=RandomForestClassifier(n_estimators=10),
                               treatment_levels=0)

    # check properties before fit
    assert dml_obj.n_rep_boot is None
    assert dml_obj.coef is None
    assert dml_obj.all_coef is None
    assert dml_obj.se is None
    assert dml_obj.all_se is None
    assert dml_obj.t_stat is None
    assert dml_obj.pval is None
    assert dml_obj.n_rep_boot is None
    assert dml_obj.boot_t_stat is None
    assert dml_obj.boot_method is None
    assert dml_obj.sensitivity_elements is None
    assert dml_obj.sensitivity_params is None

    # check properties after fit
    dml_obj.fit()
    assert dml_obj.coef is not None
    assert dml_obj.all_coef is not None
    assert dml_obj.se is not None
    assert dml_obj.all_se is not None
    assert dml_obj.t_stat is not None
    assert dml_obj.pval is not None
    assert dml_obj.n_rep_boot is None
    assert dml_obj.boot_t_stat is None
    assert dml_obj.boot_method is None
    assert dml_obj.sensitivity_elements is not None
    assert dml_obj.sensitivity_params is None

    # check properties after bootstrap
    dml_obj.bootstrap()
    assert dml_obj.n_rep_boot is not None
    assert dml_obj.boot_t_stat is not None
    assert dml_obj.boot_method is not None

    # check properties after sensitivity analysis
    dml_obj.sensitivity_analysis()
    assert dml_obj.sensitivity_params is not None


@pytest.fixture(scope='module',
                params=[[LinearRegression(),
                         LogisticRegression(solver='lbfgs', max_iter=250, random_state=42)],
                        [RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
                         RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42)]])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=[1, 5])
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
def dml_apos_fixture(learner, n_rep, normalize_ipw, trimming_threshold, treatment_levels):
    boot_methods = ['normal']
    n_folds = 2
    n_rep_boot = 499

    np.random.seed(3141)
    n_obs = 500
    data = make_irm_data_discrete_treatments(n_obs=n_obs)
    y = data['y']
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
        n_rep=n_rep,
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
    if n_rep == 1:
        for bootstrap in boot_methods:
            np.random.seed(42)
            boot_t_stat = boot_apos(res_manual['apo_scaled_score'], res_manual['all_se'], treatment_levels,
                                    all_smpls, n_rep, bootstrap, n_rep_boot)

            np.random.seed(42)
            dml_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)

            res_dict['boot_t_stat_' + bootstrap] = dml_obj.boot_t_stat
            res_dict['boot_t_stat_' + bootstrap + '_manual'] = boot_t_stat

            ci = dml_obj.confint(joint=True, level=0.95)
            ci_manual = confint_manual(
                res_manual['apos'], res_manual['se'], treatment_levels,
                boot_t_stat=boot_t_stat, joint=True, level=0.95)
            res_dict['boot_ci_' + bootstrap] = ci.to_numpy()
            res_dict['boot_ci_' + bootstrap + '_manual'] = ci_manual.to_numpy()

    # causal contrasts
    if len(treatment_levels) > 1:
        acc_single = dml_obj.causal_contrast(reference_levels=[treatment_levels[0]])
        res_dict['causal_contrast_single'] = acc_single
        acc_multiple = dml_obj.causal_contrast(reference_levels=treatment_levels)
        res_dict['causal_contrast_multiple'] = acc_multiple

    return res_dict


@pytest.mark.ci
def test_dml_apos_coef(dml_apos_fixture):
    assert np.allclose(dml_apos_fixture['coef'],
                       dml_apos_fixture['coef_manual'],
                       rtol=1e-9, atol=1e-9)
    assert np.allclose(dml_apos_fixture['coef'],
                       dml_apos_fixture['coef_ext_smpls'],
                       rtol=1e-9, atol=1e-9)


@pytest.mark.ci
def test_dml_apos_se(dml_apos_fixture):
    if dml_apos_fixture['n_rep'] != 1:
        pytest.skip("Skipping test as n_rep is not 1")
    assert np.allclose(dml_apos_fixture['se'],
                       dml_apos_fixture['se_manual'],
                       rtol=1e-9, atol=1e-9)
    assert np.allclose(dml_apos_fixture['se'],
                       dml_apos_fixture['se_ext_smpls'],
                       rtol=1e-9, atol=1e-9)


@pytest.mark.ci
def test_dml_apos_boot(dml_apos_fixture):
    if dml_apos_fixture['n_rep'] != 1:
        pytest.skip("Skipping test as n_rep is not 1")
    for bootstrap in dml_apos_fixture['boot_methods']:
        assert np.allclose(dml_apos_fixture['boot_t_stat_' + bootstrap],
                           dml_apos_fixture['boot_t_stat_' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_apos_ci(dml_apos_fixture):
    if dml_apos_fixture['n_rep'] != 1:
        pytest.skip("Skipping test as n_rep is not 1")
    for bootstrap in dml_apos_fixture['boot_methods']:
        assert np.allclose(dml_apos_fixture['ci'],
                           dml_apos_fixture['ci_manual'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_apos_fixture['ci'],
                           dml_apos_fixture['ci_ext_smpls'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_apos_fixture['boot_ci_' + bootstrap],
                           dml_apos_fixture['boot_ci_' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_doubleml_apos_return_types(dml_apos_fixture):
    assert isinstance(dml_apos_fixture['apos_model'].__str__(), str)
    assert isinstance(dml_apos_fixture['apos_model'].summary, pd.DataFrame)

    assert dml_apos_fixture['apos_model'].all_coef.shape == (
        dml_apos_fixture['n_treatment_levels'],
        dml_apos_fixture['n_rep']
    )
    assert isinstance(dml_apos_fixture['unfitted_apos_model'].summary, pd.DataFrame)
    if dml_apos_fixture['n_treatment_levels'] > 1:
        assert isinstance(dml_apos_fixture['causal_contrast_single'], dml.DoubleMLFramework)
        assert isinstance(dml_apos_fixture['causal_contrast_multiple'], dml.DoubleMLFramework)

    benchmark = dml_apos_fixture['apos_model'].sensitivity_benchmark(benchmarking_set=['x1'])
    assert isinstance(benchmark, pd.DataFrame)


@pytest.mark.ci
def test_doubleml_apos_causal_contrast(dml_apos_fixture):
    if dml_apos_fixture['n_treatment_levels'] == 1:
        pytest.skip("Skipping test as n_treatment_levels is 1")

    acc_single = dml_apos_fixture['apos_model'].all_coef[1:, ] - dml_apos_fixture['apos_model'].all_coef[0, ]
    assert np.allclose(dml_apos_fixture['causal_contrast_single'].all_thetas,
                       acc_single,
                       rtol=1e-9, atol=1e-9)

    acc_multiple = np.append(acc_single,
                             dml_apos_fixture['apos_model'].all_coef[2:3, ] - dml_apos_fixture['apos_model'].all_coef[1:2, ],
                             axis=0)
    assert np.allclose(dml_apos_fixture['causal_contrast_multiple'].all_thetas,
                       acc_multiple,
                       rtol=1e-9, atol=1e-9)
