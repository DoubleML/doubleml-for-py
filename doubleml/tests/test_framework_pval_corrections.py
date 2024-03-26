import pytest

import numpy as np

from doubleml.double_ml_framework import DoubleMLFramework
from ._utils import generate_dml_dict


@pytest.fixture(scope='module',
                params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope='module',
                params=[1, 5])
def n_thetas(request):
    return request.param


@pytest.fixture(scope='module',
                params=[0.05, 0.1, 0.2])
def sig_level(request):
    return request.param


@pytest.fixture(scope='module')
def dml_framework_tstat_pval_fixture(n_rep, n_thetas):
    n_obs = 100

    # generate score samples
    psi_a = np.ones(shape=(n_obs, n_thetas, n_rep))
    psi_b = np.random.normal(size=(n_obs, n_thetas, n_rep))
    doubleml_dict = generate_dml_dict(psi_a, psi_b)
    dml_framework_obj = DoubleMLFramework(doubleml_dict)

    result_dict = {
        'dml_framework_obj': dml_framework_obj,
    }

    return result_dict


@pytest.mark.ci
def test_dml_framework_tstat_shape(dml_framework_tstat_pval_fixture):
    dml_framework_obj = dml_framework_tstat_pval_fixture['dml_framework_obj']

    t_stats = dml_framework_obj.t_stats
    assert dml_framework_obj.t_stats.shape == (dml_framework_obj.n_thetas, )
    assert np.all(np.isfinite(t_stats))

    all_t_stats = dml_framework_obj.all_t_stats
    assert all_t_stats.shape == (dml_framework_obj.n_thetas, dml_framework_obj.n_rep)
    assert np.all(np.isfinite(all_t_stats))


@pytest.mark.ci
def test_dml_framework_pval_shape(dml_framework_tstat_pval_fixture):
    dml_framework_obj = dml_framework_tstat_pval_fixture['dml_framework_obj']

    p_vals = dml_framework_obj.pvals
    assert p_vals.shape == (dml_framework_obj.n_thetas, )
    assert np.all(np.isfinite(p_vals))

    all_p_vals = dml_framework_obj.all_pvals
    assert all_p_vals.shape == (dml_framework_obj.n_thetas, dml_framework_obj.n_rep)
    assert np.all(np.isfinite(all_p_vals))


@pytest.fixture(scope='module')
def dml_framework_pval_cov_fixture(n_rep, sig_level):
    np.random.seed(42)
    n_thetas = 10
    n_obs = 200
    repetitions = 500

    avg_type1_error_single_estimate = np.full(repetitions, np.nan)
    avg_type1_error_all_single_estimate = np.full(repetitions, np.nan)

    type1_error_bonf = np.full(repetitions, np.nan)
    type1_error_holm = np.full(repetitions, np.nan)
    type1_error_rw = np.full(repetitions, np.nan)

    for i in range(repetitions):
        # generate score samples
        psi_a = np.ones(shape=(n_obs, n_thetas, n_rep))
        psi_b = np.random.normal(size=(n_obs, n_thetas, n_rep))
        doubleml_dict = generate_dml_dict(psi_a, psi_b)
        dml_framework_obj = DoubleMLFramework(doubleml_dict)

        p_vals = dml_framework_obj.pvals
        all_p_vals = dml_framework_obj.all_pvals
        avg_type1_error_single_estimate[i] = np.mean(p_vals < sig_level)
        avg_type1_error_all_single_estimate[i] = np.mean(all_p_vals < sig_level)

        # p_value corrections
        # bonferroni
        p_vals_bonf, _ = dml_framework_obj.p_adjust(method='bonferroni')
        type1_error_bonf[i] = any(p_vals_bonf['pval'] < sig_level)

        # holm
        p_vals_holm, _ = dml_framework_obj.p_adjust(method='holm')
        type1_error_holm[i] = any(p_vals_holm['pval'] < sig_level)

        # romano-wolf
        dml_framework_obj.bootstrap(n_rep_boot=1000)
        p_vals_rw, _ = dml_framework_obj.p_adjust(method='romano-wolf')
        type1_error_rw[i] = any(p_vals_rw['pval'] < sig_level)

    result_dict = {
        'sig_level': sig_level,
        'avg_type1_error_single_estimate': np.mean(avg_type1_error_single_estimate),
        'avg_type1_error_all_single_estimate': np.mean(avg_type1_error_all_single_estimate),
        'FWER_bonf': np.mean(type1_error_bonf),
        'FWER_holm': np.mean(type1_error_holm),
        'FWER_rw': np.mean(type1_error_rw),
    }

    return result_dict


@pytest.mark.ci
def test_dml_framework_pval_FWER(dml_framework_pval_cov_fixture):
    sig_level = dml_framework_pval_cov_fixture['sig_level']
    avg_type1_error_single_estimate = dml_framework_pval_cov_fixture['avg_type1_error_single_estimate']
    avg_type1_error_all_single_estimate = dml_framework_pval_cov_fixture['avg_type1_error_all_single_estimate']

    tolerance = 0.02
    # only one-sided since median aggregation over independent data
    assert avg_type1_error_single_estimate <= sig_level + tolerance
    assert (sig_level - tolerance <= avg_type1_error_all_single_estimate) & \
        (avg_type1_error_all_single_estimate <= sig_level + tolerance)

    # test FWER control
    FWER_bonf = dml_framework_pval_cov_fixture['FWER_bonf']
    assert FWER_bonf <= sig_level + tolerance

    FWER_holm = dml_framework_pval_cov_fixture['FWER_holm']
    assert FWER_holm <= sig_level + tolerance

    FWER_rw = dml_framework_pval_cov_fixture['FWER_rw']
    assert FWER_rw <= sig_level + tolerance
