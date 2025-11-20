import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import doubleml as dml
from doubleml.plm.datasets import make_lplr_LZZ2020


@pytest.fixture(scope="module", params=[RandomForestClassifier(random_state=42, max_depth=2, n_estimators=10)])
def learner_M(request):
    return request.param


@pytest.fixture(scope="module", params=[RandomForestRegressor(random_state=42, max_depth=2, n_estimators=10)])
def learner_t(request):
    return request.param


@pytest.fixture(scope="module", params=[RandomForestRegressor(random_state=42, max_depth=2, n_estimators=10)])
def learner_m(request):
    return request.param


@pytest.fixture(scope="module", params=[RandomForestClassifier(random_state=42, max_depth=2, n_estimators=10)])
def learner_m_classifier(request):
    return request.param


@pytest.fixture(scope="module", params=["nuisance_space", "instrument"])
def score(request):
    return request.param


@pytest.fixture(scope="module", params=["continuous", "binary", "binary_unbalanced"])
def treatment(request):
    return request.param


@pytest.fixture(scope="module")
def dml_lplr_fixture(
    score,
    learner_M,
    learner_t,
    learner_m,
    learner_m_classifier,
    treatment,
):
    n_folds = 5
    alpha = 0.5

    # collect data
    np.random.seed(42)
    obj_dml_data = make_lplr_LZZ2020(alpha=alpha, treatment=treatment)

    ml_M = clone(learner_M)
    ml_t = clone(learner_t)
    if treatment == "continuous":
        ml_m = clone(learner_m)
    else:
        ml_m = clone(learner_m_classifier)

    dml_sel_obj = dml.DoubleMLLPLR(obj_dml_data, ml_M, ml_t, ml_m, n_folds=n_folds, score=score)
    dml_sel_obj.fit()

    res_dict = {
        "coef": dml_sel_obj.coef[0],
        "se": dml_sel_obj.se[0],
        "true_coef": alpha,
    }

    return res_dict


@pytest.mark.ci
def test_dml_lplr_coef(dml_lplr_fixture):
    # true_coef should lie within three standard deviations of the estimate
    coef = dml_lplr_fixture["coef"]
    se = dml_lplr_fixture["se"]
    true_coef = dml_lplr_fixture["true_coef"]
    assert abs(coef - true_coef) <= 3.0 * np.sqrt(se)
