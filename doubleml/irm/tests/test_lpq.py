import numpy as np
import pytest
import math

import doubleml as dml

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from statsmodels.nonparametric.kde import KDEUnivariate

from ...tests._utils import draw_smpls
from ._utils_lpq_manual import fit_lpq
from ...utils._estimation import _default_kde


def custom_kde(u, weights):
    dens = KDEUnivariate(u)
    dens.fit(kernel='epa', bw='silverman', weights=weights, fft=False)

    return dens.evaluate(0)


@pytest.fixture(scope='module',
                params=[0, 1])
def treatment(request):
    return request.param


@pytest.fixture(scope='module',
                params=[0.25, 0.75])
def quantile(request):
    return request.param


@pytest.fixture(scope='module',
                params=[LogisticRegression()])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=[True, False])
def normalize_ipw(request):
    return request.param


@pytest.fixture(scope='module',
                params=[0.05])
def trimming_threshold(request):
    return request.param


@pytest.fixture(scope='module',
                params=['default', custom_kde])
def kde(request):
    return request.param


@pytest.fixture(scope="module")
def dml_lpq_fixture(generate_data_local_quantiles, treatment, quantile, learner,
                    normalize_ipw, trimming_threshold, kde):
    n_folds = 3

    # collect data
    (x, y, d, z) = generate_data_local_quantiles
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d, z)
    np.random.seed(42)
    n_obs = len(y)
    strata = d + 2 * z
    all_smpls = draw_smpls(n_obs, n_folds, n_rep=1, groups=strata)

    np.random.seed(42)
    if kde == 'default':
        dml_lpq_obj = dml.DoubleMLLPQ(obj_dml_data,
                                      clone(learner), clone(learner),
                                      treatment=treatment,
                                      quantile=quantile,
                                      n_folds=n_folds,
                                      n_rep=1,
                                      normalize_ipw=normalize_ipw,
                                      trimming_threshold=trimming_threshold,
                                      draw_sample_splitting=False)
        # synchronize the sample splitting
        dml_lpq_obj.set_sample_splitting(all_smpls=all_smpls)
        dml_lpq_obj.fit()

        np.random.seed(42)
        res_manual = fit_lpq(y, x, d, z, quantile, clone(learner), clone(learner),
                             all_smpls, treatment,
                             normalize_ipw=normalize_ipw, kde=_default_kde,
                             n_rep=1, trimming_threshold=trimming_threshold)
    else:
        dml_lpq_obj = dml.DoubleMLLPQ(obj_dml_data,
                                      clone(learner), clone(learner),
                                      treatment=treatment,
                                      quantile=quantile,
                                      n_folds=n_folds,
                                      n_rep=1,
                                      normalize_ipw=normalize_ipw,
                                      kde=kde,
                                      trimming_threshold=trimming_threshold,
                                      draw_sample_splitting=False)

        # synchronize the sample splitting
        dml_lpq_obj.set_sample_splitting(all_smpls=all_smpls)
        dml_lpq_obj.fit()

        np.random.seed(42)
        res_manual = fit_lpq(y, x, d, z, quantile, clone(learner), clone(learner),
                             all_smpls, treatment,
                             normalize_ipw=normalize_ipw, kde=kde,
                             n_rep=1, trimming_threshold=trimming_threshold)

    res_dict = {'coef': dml_lpq_obj.coef[0],
                'coef_manual': res_manual['lpq'],
                'se': dml_lpq_obj.se[0],
                'se_manual': res_manual['se']}

    return res_dict


@pytest.mark.ci
def test_dml_lpq_coef(dml_lpq_fixture):
    assert math.isclose(dml_lpq_fixture['coef'],
                        dml_lpq_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_lpq_se(dml_lpq_fixture):
    assert math.isclose(dml_lpq_fixture['se'],
                        dml_lpq_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)
