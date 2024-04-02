import numpy as np
import pytest
import math

import doubleml as dml

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from ...tests._utils import draw_smpls
from ._utils_pq_manual import fit_pq


@pytest.fixture(scope='module',
                params=[0, 1])
def treatment(request):
    return request.param


@pytest.fixture(scope='module',
                params=[0.25, 0.5, 0.75])
def quantile(request):
    return request.param


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
                params=[0.01, 0.05])
def trimming_threshold(request):
    return request.param


@pytest.fixture(scope="module")
def dml_pq_fixture(generate_data_quantiles, treatment, quantile, learner,
                   normalize_ipw, trimming_threshold):
    n_folds = 3

    # collect data
    (x, y, d) = generate_data_quantiles
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d)
    np.random.seed(42)
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds, n_rep=1, groups=d)

    np.random.seed(42)
    dml_pq_obj = dml.DoubleMLPQ(obj_dml_data,
                                clone(learner), clone(learner),
                                treatment=treatment,
                                quantile=quantile,
                                n_folds=n_folds,
                                n_rep=1,
                                trimming_threshold=trimming_threshold,
                                normalize_ipw=normalize_ipw,
                                draw_sample_splitting=False)

    # synchronize the sample splitting
    dml_pq_obj.set_sample_splitting(all_smpls=all_smpls)
    np.random.seed(42)
    dml_pq_obj.fit()

    np.random.seed(42)
    res_manual = fit_pq(y, x, d, quantile,
                        clone(learner), clone(learner),
                        all_smpls, treatment,
                        n_rep=1,
                        trimming_threshold=trimming_threshold,
                        normalize_ipw=normalize_ipw)

    res_dict = {'coef': dml_pq_obj.coef,
                'coef_manual': res_manual['pq'],
                'se': dml_pq_obj.se,
                'se_manual': res_manual['se']}

    return res_dict


@pytest.mark.ci
def test_dml_pq_coef(dml_pq_fixture):
    assert math.isclose(dml_pq_fixture['coef'],
                        dml_pq_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_pq_se(dml_pq_fixture):
    assert math.isclose(dml_pq_fixture['se'],
                        dml_pq_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)
