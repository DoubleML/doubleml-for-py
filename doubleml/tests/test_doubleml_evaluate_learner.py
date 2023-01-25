import pytest
import numpy as np
import doubleml as dml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from doubleml.datasets import make_irm_data
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


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
                params=['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope='module',
                params=[mean_absolute_error, mean_squared_error])
def metric(request):
    return request.param


@pytest.fixture(scope='module',
                params=[0.01, 0.05])
def trimming_threshold(request):
    return request.param


@pytest.fixture(scope='module')
def dml_irm_eval_learner_fixture(metric, learner, dml_procedure, trimming_threshold, n_rep):
    # Set machine learning methods for m & g
    ml_g = clone(learner[0])
    ml_m = clone(learner[1])

    np.random.seed(3141)
    dml_irm_obj = dml.DoubleMLIRM(obj_dml_data,
                                  ml_g, ml_m,
                                  n_folds=2,
                                  n_rep=n_rep,
                                  dml_procedure=dml_procedure,
                                  trimming_threshold=trimming_threshold)
    dml_irm_obj.fit()
    res = dml_irm_obj.evaluate_learners(metric=metric)
    return res


@pytest.mark.ci
def test_dml_irm_eval_learner(dml_irm_eval_learner_fixture, n_rep):
    assert dml_irm_eval_learner_fixture['ml_g0'].shape == (n_rep, 1)
    assert dml_irm_eval_learner_fixture['ml_g1'].shape == (n_rep, 1)
    assert dml_irm_eval_learner_fixture['ml_m'].shape == (n_rep, 1)
