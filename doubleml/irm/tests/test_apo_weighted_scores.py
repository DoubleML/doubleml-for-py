import pytest
import numpy as np

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ...tests._utils import draw_smpls
import doubleml as dml


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
                params=[0, 1])
def treatment_level(request):
    return request.param


@pytest.fixture(scope='module')
def weighted_apo_score_fixture(generate_data_irm, learner, score, n_rep, normalize_ipw, trimming_threshold,
                               treatment_level):
    n_folds = 2

    # collect data
    (x, y, d) = generate_data_irm
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds, n_rep=1, groups=d)
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d)

    input_args = {
        "obj_dml_data": obj_dml_data,
        "ml_g": clone(learner[0]),
        "ml_m": clone(learner[1]),
        "treatment_level": treatment_level,
        "n_folds": n_folds,
        "n_rep": n_rep,
        "score": score,
        "normalize_ipw": normalize_ipw,
        "trimming_threshold": trimming_threshold,
        "draw_sample_splitting": False,
    }

    np.random.seed(42)
    dml_obj = dml.DoubleMLAPO(**input_args)
    dml_obj.set_sample_splitting(all_smpls=all_smpls)
    dml_obj.fit()

    np.random.seed(42)
    weights = 0.5 * np.ones_like(obj_dml_data.y)
    dml_obj_weighted = dml.DoubleMLAPO(weights=weights, **input_args)
    dml_obj_weighted.set_sample_splitting(all_smpls=all_smpls)
    dml_obj_weighted.fit()

    np.random.seed(42)
    weights_dict = {
        'weights': weights,
        'weights_bar': np.tile(weights[:, np.newaxis], (1, n_rep)),
    }
    dml_obj_weighted_dict = dml.DoubleMLAPO(weights=weights_dict, **input_args)
    dml_obj_weighted_dict.set_sample_splitting(all_smpls=all_smpls)
    dml_obj_weighted_dict.fit()

    result_dict = {
        'coef': dml_obj.coef,
        'weighted_coef': dml_obj_weighted.coef,
        'weighted_coef_dict': dml_obj_weighted_dict.coef,
        'default_weights': dml_obj.weights,
    }
    return result_dict


@pytest.mark.ci
def test_apo_weighted_coef(weighted_apo_score_fixture):
    assert np.allclose(0.5 * weighted_apo_score_fixture['coef'],
                       weighted_apo_score_fixture['weighted_coef'])
    assert np.allclose(0.5 * weighted_apo_score_fixture['coef'],
                       weighted_apo_score_fixture['weighted_coef_dict'])


@pytest.mark.ci
def test_apo_default_weights(weighted_apo_score_fixture):
    assert isinstance(weighted_apo_score_fixture['default_weights'], dict)

    expected_keys = {'weights'}
    assert set(weighted_apo_score_fixture['default_weights'].keys()) == expected_keys

    assert np.allclose(weighted_apo_score_fixture['default_weights']['weights'],
                       np.ones_like(weighted_apo_score_fixture['default_weights']['weights']))
