import numpy as np
import pytest
import math

import doubleml as dml

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ...tests._utils import draw_smpls
from ._utils_cvar_manual import fit_cvar


@pytest.fixture(scope='module',
                params=[0, 1])
def treatment(request):
    return request.param


@pytest.fixture(scope='module',
                params=[0.25, 0.5, 0.75])
def quantile(request):
    return request.param


@pytest.fixture(scope='module',
                params=[[LinearRegression(),
                         LogisticRegression(solver='lbfgs', max_iter=250)],
                        [RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
                         RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42)]])
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
def dml_cvar_fixture(generate_data_quantiles, treatment, quantile, learner,
                     normalize_ipw, trimming_threshold):
    n_folds = 3

    # Set machine learning methods for m & g
    ml_g = clone(learner[0])
    ml_m = clone(learner[1])

    # collect data
    (x, y, d) = generate_data_quantiles
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d)
    np.random.seed(42)
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds, n_rep=1, groups=d)

    np.random.seed(42)
    dml_cvar_obj = dml.DoubleMLCVAR(obj_dml_data,
                                    clone(ml_g), clone(ml_m),
                                    treatment=treatment,
                                    quantile=quantile,
                                    n_folds=n_folds,
                                    n_rep=1,
                                    normalize_ipw=normalize_ipw,
                                    trimming_threshold=trimming_threshold,
                                    draw_sample_splitting=False)

    # synchronize the sample splitting
    dml_cvar_obj.set_sample_splitting(all_smpls=all_smpls)
    dml_cvar_obj.fit()

    np.random.seed(42)
    res_manual = fit_cvar(y, x, d, quantile,
                          clone(ml_g), clone(ml_m),
                          all_smpls, treatment,
                          normalize_ipw=normalize_ipw,
                          n_rep=1, trimming_threshold=trimming_threshold)

    res_dict = {'coef': dml_cvar_obj.coef,
                'coef_manual': res_manual['pq'],
                'se': dml_cvar_obj.se,
                'se_manual': res_manual['se']}

    return res_dict


@pytest.mark.ci
def test_dml_cvar_coef(dml_cvar_fixture):
    assert math.isclose(dml_cvar_fixture['coef'],
                        dml_cvar_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_cvar_se(dml_cvar_fixture):
    assert math.isclose(dml_cvar_fixture['se'],
                        dml_cvar_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)
