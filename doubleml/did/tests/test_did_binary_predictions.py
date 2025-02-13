import math

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml.did import DoubleMLDIDBinary
from doubleml.did.datasets import make_did_SZ2020
from doubleml.utils import DMLDummyClassifier, DMLDummyRegressor

from ...tests._utils import draw_smpls


@pytest.fixture(scope="module", params=["observational", "experimental"])
def did_score(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module")
def doubleml_did_fixture(did_score, n_rep):
    n_obs = 500
    n_folds = 5

    ext_predictions = {"d": {}}
    dml_data = make_did_SZ2020(n_obs=n_obs, return_type="DoubleMLPanelData")

    kwargs = {
        "obj_dml_data": dml_data,
        "g_value": 1,
        "t_value_pre": 0,
        "t_value_eval": 1,
        "score": did_score,
        "n_rep": n_rep,
        "draw_sample_splitting": False,
    }

    dml_did = DoubleMLDIDBinary(ml_g=LinearRegression(), ml_m=LogisticRegression(), **kwargs)
    all_smpls = draw_smpls(n_obs, n_folds, n_rep=n_rep, groups=dml_did._g_panel)
    dml_did.set_sample_splitting(all_smpls)

    np.random.seed(3141)
    dml_did.fit(store_predictions=True)

    ext_predictions["d"]["ml_g0"] = dml_did.predictions["ml_g0"][:, :, 0]
    ext_predictions["d"]["ml_g1"] = dml_did.predictions["ml_g1"][:, :, 0]
    if did_score == "observational":
        ext_predictions["d"]["ml_m"] = dml_did.predictions["ml_m"][:, :, 0]

    dml_did_ext = DoubleMLDIDBinary(ml_g=DMLDummyRegressor(), ml_m=DMLDummyClassifier(), **kwargs)
    dml_did_ext.set_sample_splitting(all_smpls)
    np.random.seed(3141)
    dml_did_ext.fit(external_predictions=ext_predictions)

    res_dict = {
        "coef": dml_did.coef[0],
        "coef_ext": dml_did_ext.coef[0],
        "se": dml_did.se[0],
        "se_ext": dml_did_ext.se[0],
        "score": dml_did.psi,
        "score_ext": dml_did_ext.psi,
    }

    return res_dict


@pytest.mark.ci
def test_coef(doubleml_did_fixture):
    assert math.isclose(doubleml_did_fixture["coef"], doubleml_did_fixture["coef_ext"], rel_tol=1e-9, abs_tol=1e-3)


@pytest.mark.ci
def test_se(doubleml_did_fixture):
    assert math.isclose(doubleml_did_fixture["se"], doubleml_did_fixture["se_ext"], rel_tol=1e-9, abs_tol=1e-3)


@pytest.mark.ci
def test_score(doubleml_did_fixture):
    assert np.allclose(doubleml_did_fixture["score"], doubleml_did_fixture["score_ext"], rtol=1e-9, atol=1e-3)
