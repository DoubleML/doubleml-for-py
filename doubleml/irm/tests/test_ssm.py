import pytest
import math
import numpy as np

from sklearn.base import clone

from sklearn.linear_model import LassoCV, LogisticRegressionCV

import doubleml as dml

from ...tests._utils import draw_smpls
from ._utils_ssm_manual import fit_selection


@pytest.fixture(scope='module',
                params=[[LassoCV(),
                         LogisticRegressionCV(penalty='l1', solver='liblinear')]])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['missing-at-random', 'nonignorable'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=[True, False])
def normalize_ipw(request):
    return request.param


@pytest.fixture(scope='module',
                params=[0.01])
def trimming_threshold(request):
    return request.param


@pytest.fixture(scope='module')
def dml_selection_fixture(generate_data_selection_mar, generate_data_selection_nonignorable,
                          learner, score,
                          trimming_threshold, normalize_ipw):
    n_folds = 3

    # collect data
    np.random.seed(42)
    if score == 'missing-at-random':
        (x, y, d, z, s) = generate_data_selection_mar
    else:
        (x, y, d, z, s) = generate_data_selection_nonignorable

    ml_g = clone(learner[0])
    ml_pi = clone(learner[1])
    ml_m = clone(learner[1])

    np.random.seed(42)
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds)

    np.random.seed(42)
    if score == 'missing-at-random':
        obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d, z=None, s=s)
        dml_sel_obj = dml.DoubleMLSSM(obj_dml_data,
                                      ml_g, ml_pi, ml_m,
                                      n_folds=n_folds,
                                      score=score)
    else:
        assert score == 'nonignorable'
        obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d, z=z, s=s)
        dml_sel_obj = dml.DoubleMLSSM(obj_dml_data,
                                      ml_g, ml_pi, ml_m,
                                      n_folds=n_folds,
                                      score=score)

    np.random.seed(42)
    dml_sel_obj.set_sample_splitting(all_smpls=all_smpls)
    dml_sel_obj.fit()

    np.random.seed(42)
    res_manual = fit_selection(y, x, d, z, s,
                               clone(learner[0]), clone(learner[1]), clone(learner[1]),
                               all_smpls, score,
                               trimming_rule='truncate',
                               trimming_threshold=trimming_threshold,
                               normalize_ipw=normalize_ipw)

    res_dict = {'coef': dml_sel_obj.coef[0],
                'coef_manual': res_manual['theta'],
                'se': dml_sel_obj.se[0],
                'se_manual': res_manual['se']}

    # sensitivity tests
    # TODO

    return res_dict


@pytest.mark.ci
def test_dml_selection_coef(dml_selection_fixture):
    assert math.isclose(dml_selection_fixture['coef'],
                        dml_selection_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-2)


@pytest.mark.ci
def test_dml_selection_se(dml_selection_fixture):
    assert math.isclose(dml_selection_fixture['se'],
                        dml_selection_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=5e-2)
