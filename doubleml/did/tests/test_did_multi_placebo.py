import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml.data import DoubleMLPanelData
from doubleml.did import DoubleMLDIDMulti
from doubleml.did.datasets import make_did_CS2021


@pytest.fixture(scope="module", params=["observational", "experimental"])
def did_score(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module")
def doubleml_did_fixture(did_score, n_rep):
    n_obs = 1000
    dgp = 5  # has to be experimental (for experimental score to be valid)
    np.random.seed(42)
    df = make_did_CS2021(n_obs=n_obs, dgp=dgp, n_pre_treat_periods=3, n_periods=5, time_type="float")
    dml_data = DoubleMLPanelData(df, y_col="y", d_cols="d", t_col="t", id_col="id", x_cols=["Z1", "Z2", "Z3", "Z4"])

    # all placebo combinations
    gt_combinations_group3 = [(3, 0, 1), (3, 0, 2), (3, 1, 2)]
    gt_combinations_group4 = [(4, 0, 1), (4, 0, 2), (4, 0, 3), (4, 1, 2), (4, 1, 3), (4, 2, 3)]
    gt_combinations = gt_combinations_group3 + gt_combinations_group4

    kwargs = {
        "obj_dml_data": dml_data,
        "ml_g": LinearRegression(),
        "ml_m": LogisticRegression(),
        "gt_combinations": gt_combinations,
        "score": did_score,
        "n_rep": n_rep,
        "n_folds": 5,
        "draw_sample_splitting": True,
    }

    dml_did = DoubleMLDIDMulti(**kwargs)

    np.random.seed(3141)
    dml_did.fit()
    ci = dml_did.confint(level=0.95)

    res_dict = {
        "coef": dml_did.coef[:],
        "ci_lower": ci.iloc[:, 0],
        "ci_upper": ci.iloc[:, 1],
    }

    return res_dict


@pytest.mark.ci
def test_zero(doubleml_did_fixture):
    assert all(doubleml_did_fixture["ci_lower"] <= 0.0)
    assert all(doubleml_did_fixture["ci_upper"] >= 0.0)
