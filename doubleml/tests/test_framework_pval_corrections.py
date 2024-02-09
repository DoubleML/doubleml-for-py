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


