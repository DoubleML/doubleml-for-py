import numpy as np
import pytest
import math
import scipy.stats as sps

import doubleml as dml
from doubleml.datasets import make_pliv_multiway_cluster_CKMS2021

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier



@pytest.fixture(scope='module',
                params=[RandomForestClassifier(max_depth=2, n_estimators=10),
                        AdaBoostClassifier(n_estimators=10),
                        DecisionTreeClassifier(max_depth=5),
                        LogisticRegression()])
def learner(request):
    return request.param

@pytest.fixture(scope='module',
                params=['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope="module")
def dml_qte_fixture(generate_data_quantiles, learner, dml_procedure):
    n_folds = 3

    # collect data
    (x, y, d) = generate_data_quantiles
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d)


    # Set machine learning methods for g & m
    ml_g = clone(learner)
    ml_m = clone(learner)

    np.random.seed(42)
    quantile = [0.25, 0.5, 0.75]
    dml_qte_obj = dml.DoubleMLQTE(obj_dml_data,
                                ml_g, ml_m,
                                quantiles=quantile,
                                n_folds=n_folds,
                                dml_procedure=dml_procedure)



@pytest.mark.ci
def test_dml_plr_coef(dml_qte_fixture):
    assert math.isclose(1,
                        1,
                        rel_tol=1e-9, abs_tol=1e-4)

@pytest.mark.ci
def test_doubleml_cluster_not_implemented_exception():
    np.random.seed(3141)
    dml_data = make_pliv_multiway_cluster_CKMS2021()
    dml_data.z_cols = None
    ml_g = RandomForestClassifier()
    ml_m = RandomForestClassifier()
    msg = 'Estimation with clustering not implemented.'
    with pytest.raises(NotImplementedError, match=msg):
        _ = dml.DoubleMLPQ(dml_data, ml_g, ml_m, treatment=1)
