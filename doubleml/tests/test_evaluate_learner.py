import pytest
import numpy as np
import doubleml as dml
from doubleml.datasets import make_irm_data
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from doubleml.utils._estimation import _logloss


np.random.seed(3141)
data = make_irm_data(theta=0.5, n_obs=200, dim_x=5, return_type='DataFrame')
obj_dml_data = dml.DoubleMLData(data, 'y', 'd')


@pytest.fixture(scope='module',
                params=[[LinearRegression(),
                         LogisticRegression(solver='lbfgs', max_iter=250)],
                        [RandomForestRegressor(max_depth=2, n_estimators=10),
                         RandomForestClassifier(max_depth=2, n_estimators=10)]])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=[1, 5])
def n_rep(request):
    return request.param


@pytest.fixture(scope='module',
                params=[0.01, 0.05])
def trimming_threshold(request):
    return request.param


@pytest.fixture(scope='module')
def dml_irm_eval_learner_fixture(learner, trimming_threshold, n_rep):
    # Set machine learning methods for m & g
    ml_g = clone(learner[0])
    ml_m = clone(learner[1])

    np.random.seed(3141)
    dml_irm_obj = dml.DoubleMLIRM(obj_dml_data,
                                  ml_g, ml_m,
                                  n_folds=2,
                                  n_rep=n_rep,
                                  trimming_threshold=trimming_threshold)
    dml_irm_obj.fit()
    res_manual = dml_irm_obj.evaluate_learners(learners=['ml_g0', 'ml_g1'])
    res_manual['ml_m'] = dml_irm_obj.evaluate_learners(learners=['ml_m'], metric=_logloss)['ml_m']

    res_dict = {'nuisance_loss': dml_irm_obj.nuisance_loss,
                'nuisance_loss_manual': res_manual
                }
    return res_dict


@pytest.mark.ci
def test_dml_irm_eval_learner(dml_irm_eval_learner_fixture, n_rep):
    assert dml_irm_eval_learner_fixture['nuisance_loss_manual']['ml_g0'].shape == (n_rep, 1)
    assert dml_irm_eval_learner_fixture['nuisance_loss_manual']['ml_g1'].shape == (n_rep, 1)
    assert dml_irm_eval_learner_fixture['nuisance_loss_manual']['ml_m'].shape == (n_rep, 1)

    assert np.allclose(dml_irm_eval_learner_fixture['nuisance_loss_manual']['ml_g0'],
                       dml_irm_eval_learner_fixture['nuisance_loss']['ml_g0'],
                       rtol=1e-9, atol=1e-4)
    assert np.allclose(dml_irm_eval_learner_fixture['nuisance_loss_manual']['ml_g1'],
                       dml_irm_eval_learner_fixture['nuisance_loss']['ml_g1'],
                       rtol=1e-9, atol=1e-4)
    assert np.allclose(dml_irm_eval_learner_fixture['nuisance_loss_manual']['ml_m'],
                       dml_irm_eval_learner_fixture['nuisance_loss']['ml_m'],
                       rtol=1e-9, atol=1e-4)
