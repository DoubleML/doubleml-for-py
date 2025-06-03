import math

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml.did import DoubleMLDIDCSBinary
from doubleml.did.datasets import make_did_SZ2020
from doubleml.tests._utils import draw_smpls
from doubleml.utils import DMLDummyClassifier, DMLDummyRegressor


@pytest.fixture(scope="module", params=["observational", "experimental"])
def did_score(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module")
def doubleml_did_cs_fixture(did_score, n_rep):
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

    dml_did = DoubleMLDIDCSBinary(ml_g=LinearRegression(), ml_m=LogisticRegression(), **kwargs)
    strata = dml_did.data_subset["G_indicator"] + 2 * dml_did.data_subset["t_indicator"]
    all_smpls = draw_smpls(2 * n_obs, n_folds, n_rep=n_rep, groups=strata)
    dml_did.set_sample_splitting(all_smpls)

    np.random.seed(3141)
    dml_did.fit(store_predictions=True)

    all_keys = ["ml_g_d0_t0", "ml_g_d0_t1", "ml_g_d1_t0", "ml_g_d1_t1"]
    for key in all_keys:
        ext_predictions["d"][key] = dml_did.predictions[key][:, :, 0]
    if did_score == "observational":
        ext_predictions["d"]["ml_m"] = dml_did.predictions["ml_m"][:, :, 0]

    dml_did_ext = DoubleMLDIDCSBinary(ml_g=DMLDummyRegressor(), ml_m=DMLDummyClassifier(), **kwargs)
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
        "dml_did_nuisance_loss": dml_did.nuisance_loss,
        "dml_did_ext_nuisance_loss": dml_did_ext.nuisance_loss,
    }

    return res_dict


@pytest.mark.ci
def test_coef(doubleml_did_cs_fixture):
    assert math.isclose(doubleml_did_cs_fixture["coef"], doubleml_did_cs_fixture["coef_ext"], rel_tol=1e-9, abs_tol=1e-3)


@pytest.mark.ci
def test_se(doubleml_did_cs_fixture):
    assert math.isclose(doubleml_did_cs_fixture["se"], doubleml_did_cs_fixture["se_ext"], rel_tol=1e-9, abs_tol=1e-3)


@pytest.mark.ci
def test_score(doubleml_did_cs_fixture):
    assert np.allclose(doubleml_did_cs_fixture["score"], doubleml_did_cs_fixture["score_ext"], rtol=1e-9, atol=1e-3)


@pytest.mark.ci
def test_nuisance_loss(doubleml_did_cs_fixture):
    for key, value in doubleml_did_cs_fixture["dml_did_nuisance_loss"].items():
        assert np.allclose(value, doubleml_did_cs_fixture["dml_did_ext_nuisance_loss"][key], rtol=1e-9, atol=1e-3)
