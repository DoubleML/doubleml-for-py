import numpy as np
import pandas as pd
import pytest
import math

from sklearn.base import clone

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import doubleml as dml
from doubleml.datasets import make_irm_data_discrete_treatements
from doubleml.utils.resampling import DoubleMLResampling

from ...tests._utils import draw_smpls
from ._utils_apo_manual import fit_apo


@pytest.fixture(scope='module',
                params=[[LinearRegression(),
                         LogisticRegression(solver='lbfgs', max_iter=250)],
                        [RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
                         RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42)]])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=[False, True])
def normalize_ipw(request):
    return request.param


@pytest.fixture(scope='module',
                params=[0.2, 0.15])
def trimming_threshold(request):
    return request.param


@pytest.fixture(scope='module')
def dml_apo_fixture(generate_data_irm, learner, normalize_ipw, trimming_threshold):
    boot_methods = ['normal']
    n_folds = 2
    n_rep_boot = 499


    # Set machine learning methods for m & g
    ml_g = clone(learner[0])
    ml_m = clone(learner[1])

    np.random.seed(3141)
    n_obs = 100
    data_apo = make_irm_data_discrete_treatements(n_obs=n_obs)
    y = data_apo['y']
    x = data_apo['x']
    d = data_apo['d']
    df_apo = pd.DataFrame(
        np.column_stack((y, d, x)),
        columns=['y', 'd'] + ['x' + str(i) for i in range(data_apo['x'].shape[1])]
    )

    dml_data = dml.DoubleMLData(df_apo, 'y', 'd')
    all_smpls = draw_smpls(n_obs, n_folds, n_rep=1, groups=d)

    np.random.seed(3141)
    dml_obj = dml.DoubleMLAPO(dml_data,
                                  ml_g, ml_m,
                                  treatment_level=0,
                                  n_folds=n_folds,
                                  normalize_ipw=normalize_ipw,
                                  draw_sample_splitting=False,
                                  trimming_threshold=trimming_threshold)

    # synchronize the sample splitting
    dml_obj.set_sample_splitting(all_smpls=all_smpls)
    dml_obj.fit()

    np.random.seed(3141)
    res_manual = fit_apo(y, x, d,
                         clone(learner[0]), clone(learner[1]),
                         treatment_level=0,
                         all_smpls=all_smpls,
                         score='APO',
                         normalize_ipw=normalize_ipw,
                         trimming_threshold=trimming_threshold)

    res_dict = {'coef': dml_obj.coef,
                'coef_manual': res_manual['theta'],
                'coef_ext': dml_obj.coef_extern,
                'se': dml_obj.se,
                'se_manual': res_manual['se']}

    return res_dict


@pytest.mark.ci
def test_dml_apo_coef(dml_apo_fixture):
    assert math.isclose(dml_apo_fixture['coef'][0],
                        dml_apo_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)
    assert math.isclose(dml_apo_fixture['coef'][0],
                        dml_apo_fixture['coef_ext'][0],
                        rel_tol=1e-9, abs_tol=1e-4)