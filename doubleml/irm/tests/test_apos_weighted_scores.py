import pytest
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import doubleml as dml
from doubleml.datasets import make_irm_data_discrete_treatments


@pytest.fixture(scope='module',
                params=[[LinearRegression(),
                         LogisticRegression(solver='lbfgs', max_iter=250)],
                        [RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
                         RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42)]])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['APO'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=[1, 3])
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
def weighted_apos_score_fixture(learner, score, n_rep, normalize_ipw, trimming_threshold,
                                treatment_levels):
    n_obs = 500
    n_folds = 2

    # collect data
    data = make_irm_data_discrete_treatments(n_obs=n_obs)
    y = data['y']
    x = data['x']
    d = data['d']
    df = pd.DataFrame(
        np.column_stack((y, d, x)),
        columns=['y', 'd'] + ['x' + str(i) for i in range(data['x'].shape[1])]
    )

    obj_dml_data = dml.DoubleMLData(df, 'y', 'd')

    input_args = {
        'obj_dml_data': obj_dml_data,
        'ml_g': clone(learner[0]),
        'ml_m': clone(learner[1]),
        'treatment_levels': treatment_levels,
        'n_folds': n_folds,
        'n_rep': n_rep,
        'score': score,
        'normalize_ipw': normalize_ipw,
        'trimming_threshold': trimming_threshold,
        'trimming_rule': 'truncate'
    }

    np.random.seed(42)
    dml_obj = dml.DoubleMLAPOS(**input_args)
    dml_obj.fit()

    np.random.seed(42)
    weights = 0.5 * np.ones_like(obj_dml_data.y)
    dml_obj_weighted = dml.DoubleMLAPOS(draw_sample_splitting=False,
                                        weights=weights,
                                        **input_args)
    dml_obj_weighted.set_sample_splitting(all_smpls=dml_obj.smpls)
    dml_obj_weighted.fit()

    np.random.seed(42)
    weights_dict = {
        'weights': weights,
        'weights_bar': np.tile(weights[:, np.newaxis], (1, n_rep)),
    }
    dml_obj_weighted_dict = dml.DoubleMLAPOS(draw_sample_splitting=False,
                                             weights=weights_dict,
                                             **input_args)
    dml_obj_weighted_dict.set_sample_splitting(all_smpls=dml_obj.smpls)
    dml_obj_weighted_dict.fit()

    result_dict = {
        'coef': dml_obj.coef,
        'weighted_coef': dml_obj_weighted.coef,
        'weighted_coef_dict': dml_obj_weighted_dict.coef,
        'default_weights': dml_obj.weights,
    }
    return result_dict


@pytest.mark.ci
def test_apos_weighted_coef(weighted_apos_score_fixture):
    assert np.allclose(0.5 * weighted_apos_score_fixture['coef'],
                       weighted_apos_score_fixture['weighted_coef'])
    assert np.allclose(0.5 * weighted_apos_score_fixture['coef'],
                       weighted_apos_score_fixture['weighted_coef_dict'])


@pytest.mark.ci
def test_apos_default_weights(weighted_apos_score_fixture):
    assert isinstance(weighted_apos_score_fixture['default_weights'], np.ndarray)

    assert np.allclose(weighted_apos_score_fixture['default_weights'],
                       np.ones_like(weighted_apos_score_fixture['default_weights']))
