import pytest
import math
import numpy as np

from sklearn.base import clone

from sklearn.linear_model import LassoCV, LogisticRegressionCV

import doubleml as dml

from ._utils import draw_smpls
from ._utils_selection_manual import fit_selection

@pytest.fixture(scope='module',
                params=[[LassoCV(),
                         LogisticRegressionCV(),
                         LogisticRegressionCV()]])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['mar', 'nonignorable'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope='module',
                params=[True, False])
def normalize_ipw(request):
    return request.param


@pytest.fixture(scope='module',
                params=[0.1])
def trimming_threshold(request):
    return request.param


@pytest.fixture(scope='module')
def dml_selection_fixture(generate_data_selection, learner, score, dml_procedure,
                          trimming_threshold, normalize_ipw):
    boot_methods = ['normal']
    n_folds = 3
    n_rep_boot = 499

    # collect data
    (x, y, d, z, s) = generate_data_selection

    ml_mu = clone(learner[0])
    ml_pi = clone(learner[1])
    ml_p = clone(learner[1])

    np.random.seed(3141)
    
    if score == 'mar':
        obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d, z=None, t=s)
        dml_sel_obj = dml.DoubleMLS(obj_dml_data,
                                        ml_mu, ml_pi, ml_p,
                                        n_folds=n_folds,
                                        score=score,
                                        dml_procedure=dml_procedure)
    else:
        assert score == 'nonignorable'
        obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d, z=z, t=s)
        dml_sel_obj = dml.DoubleMLS(obj_dml_data,
                                        ml_mu, ml_pi, ml_p,
                                        n_folds=n_folds,
                                        score=score,
                                        dml_procedure=dml_procedure)

    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds)

    np.random.seed(3141)
    dml_sel_obj.set_sample_splitting(all_smpls=all_smpls)
    dml_sel_obj.fit()

    res_manual = fit_selection(y, x, d, z, s,
                          ml_mu, ml_pi, ml_p,
                          all_smpls, dml_procedure, score,
                          trimming_rule='truncate',
                          trimming_threshold=trimming_threshold,
                          normalize_ipw=normalize_ipw)

    res_dict = {'coef': dml_sel_obj.coef[0],
                'coef_manual': res_manual['theta'],
                'se': dml_sel_obj.se[0],
                'se_manual': res_manual['se'],
                'boot_methods': boot_methods}
    
    # sensitivity tests
    # TODO

    return res_dict


@pytest.mark.ci
def test_dml_selection_coef(dml_selection_fixture):
    assert math.isclose(dml_selection_fixture['coef'],
                        dml_selection_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=0.1)


@pytest.mark.ci
def test_dml_selection_se(dml_selection_fixture):
    assert math.isclose(dml_selection_fixture['se'],
                        dml_selection_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=0.1)