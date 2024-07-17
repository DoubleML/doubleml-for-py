import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import doubleml as dml
from doubleml.datasets import make_irm_data_discrete_treatments

from ...tests._utils import draw_smpls
from ._utils_apos_manual import fit_apos


@pytest.fixture(scope='module',
                params=[[LinearRegression(),
                         LogisticRegression(solver='lbfgs', max_iter=250, random_state=42)],
                        [RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
                         RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42)]])
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
def dml_apos_fixture(generate_data_irm, learner, n_rep, normalize_ipw, trimming_threshold, treatment_levels):
    boot_methods = ['normal']
    n_folds = 2
    n_rep_boot = 499

    # Set machine learning methods for m & g
    ml_g = clone(learner[0])
    ml_m = clone(learner[1])

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
    all_smpls = draw_smpls(n_obs, n_folds, n_rep=1, groups=d)

    np.random.seed(3141)
    dml_obj = dml.DoubleMLAPOS(
        dml_data,
        ml_g, ml_m,
        treatment_levels=treatment_levels,
        n_folds=n_folds,
        n_rep=n_rep,
        score='APO',
        normalize_ipw=normalize_ipw,
        trimming_rule='truncate',
        trimming_threshold=trimming_threshold,
        draw_sample_splitting=False)

    # synchronize the sample splitting
    dml_obj.set_sample_splitting(all_smpls)
    dml_obj.fit()

    np.random.seed(3141)
    res_manual = fit_apos(
        y, x, d,
        clone(learner[0]), clone(learner[1]),
        treatment_levels=treatment_levels,
        all_smpls=all_smpls,
        score='APO',
        trimming_rule='truncate',
        normalize_ipw=normalize_ipw,
        trimming_threshold=trimming_threshold)

    res_dict = {'coef': dml_obj.coef,
                'coef_manual': res_manual['apos'],
                'se': dml_obj.se,
                'se_manual': res_manual['se']}
    return res_dict


@pytest.mark.ci
def test_dml_apos_coef(dml_apos_fixture):
    assert np.allclose(dml_apos_fixture['coef'],
                       dml_apos_fixture['coef_manual'],
                       rtol=1e-9, atol=1e-9)
