import numpy as np
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor

from doubleml.data import DoubleMLPanelData
from doubleml.did import DoubleMLDIDBinary
from doubleml.did.datasets import make_did_CS2021


@pytest.fixture(scope="module", params=["observational", "experimental"])
def did_score(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module")
def doubleml_did_fixture(did_score, n_rep):
    n_obs = 500
    dgp = 5  # has to be experimental (for experimental score to be valid)
    df = make_did_CS2021(n_obs=n_obs, dgp=dgp, n_pre_treat_periods=3)
    dml_data = DoubleMLPanelData(df, y_col="y", d_cols="d", t_col="t", id_col="id", x_cols=["Z1", "Z2", "Z3", "Z4"])

    kwargs = {
        "obj_dml_data": dml_data,
        "g_value": dml_data.g_values[0],
        "t_value_pre": dml_data.t_values[0],
        "t_value_eval": dml_data.t_values[1],
        "ml_g": LGBMRegressor(verbose=-1),
        "ml_m": LGBMClassifier(verbose=-1),
        "score": did_score,
        "n_rep": n_rep,
        "n_folds": 5,
        "draw_sample_splitting": True,
    }

    dml_did = DoubleMLDIDBinary(**kwargs)

    np.random.seed(3141)
    dml_did.fit()
    ci = dml_did.confint(level=0.99)

    res_dict = {
        "coef": dml_did.coef[0],
        "ci_lower": ci.iloc[0, 0],
        "ci_upper": ci.iloc[0, 1],
    }

    return res_dict


@pytest.mark.ci
def test_zero(doubleml_did_fixture):
    assert doubleml_did_fixture["ci_lower"] <= 0.0
    assert doubleml_did_fixture["ci_upper"] >= 0.0
