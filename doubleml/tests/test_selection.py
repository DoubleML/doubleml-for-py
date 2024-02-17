import pytest
import math
import scipy
import numpy as np
import pandas as pd

from sklearn.base import clone

from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import doubleml as dml
from doubleml.double_ml_selection import DoubleMLS  ## should not be necessary

from ._utils import draw_smpls
from ._utils_plr_manual import fit_plr, plr_dml1, plr_dml2, boot_plr, fit_sensitivity_elements_plr

@pytest.fixture(scope='module',
                params=[[LinearRegression(),
                         LogisticRegression(solver='lbfgs', max_iter=250),
                         LogisticRegression(solver='lbfgs', max_iter=250)],
                         [LassoCV(),
                         LogisticRegression(solver='lbfgs', max_iter=250),
                         LogisticRegression(solver='lbfgs', max_iter=250)],
                        [RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
                         RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42),
                         RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42)]])
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
def dml_selection_fixture(generate_data_selection, learner, score, dml_procedure):
    boot_methods = ['normal']
    n_folds = 3
    n_rep_boot = 499

    # collect data
    (x, y, d, z, s) = generate_data_selection

    # Set machine learning methods for m & g
    ml_mu = clone(learner[0])
    ml_pi = clone(learner[1])
    ml_p = clone(learner[1])

    np.random.seed(3141)

    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d, z=z, t=s)
    
    if score == 'mar':
        dml_sel_obj = DoubleMLS(obj_dml_data,
                                        ml_mu, ml_pi, ml_p,
                                        n_folds=n_folds,
                                        score=score,
                                        dml_procedure=dml_procedure)
    else:
        assert score == 'nonignorable'
        dml_sel_obj = DoubleMLS(obj_dml_data,
                                        ml_mu, ml_pi, ml_p,
                                        n_folds=n_folds,
                                        score=score,
                                        dml_procedure=dml_procedure)

    dml_sel_obj.fit()


@pytest.mark.ci
def test_dml_selection_coef(dml_selection_fixture):
    assert math.isclose(dml_selection_fixture['coef'],
                        dml_selection_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_selection_se(dml_selection_fixture):
    assert math.isclose(dml_selection_fixture['se'],
                        dml_selection_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)