import numpy as np
import pytest
import math
from sklearn.linear_model import LinearRegression, LogisticRegression
from doubleml import DoubleMLDID
from doubleml.datasets import make_did_SZ2020
from doubleml.utils import DMLDummyRegressor, DMLDummyClassifier
from ...tests._utils import draw_smpls


@pytest.fixture(scope="module", params=["observational", "experimental"])
def did_score(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module")
def doubleml_did_fixture(did_score, n_rep):
    ext_predictions = {"d": {}}
    dml_data = make_did_SZ2020(n_obs=500, return_type="DoubleMLData")
    all_smpls = draw_smpls(len(dml_data.y), 5, n_rep=n_rep, groups=dml_data.d)
    kwargs = {
        "obj_dml_data": dml_data,
        "score": did_score,
        "n_rep": n_rep,
        "draw_sample_splitting": False
    }
    dml_did = DoubleMLDID(ml_g=LinearRegression(), ml_m=LogisticRegression(), **kwargs)
    dml_did.set_sample_splitting(all_smpls)
    np.random.seed(3141)
    dml_did.fit(store_predictions=True)

    ext_predictions["d"]["ml_g0"] = dml_did.predictions["ml_g0"][:, :, 0]
    ext_predictions["d"]["ml_g1"] = dml_did.predictions["ml_g1"][:, :, 0]
    ext_predictions["d"]["ml_m"] = dml_did.predictions["ml_m"][:, :, 0]

    dml_did_ext = DoubleMLDID(ml_g=DMLDummyRegressor(), ml_m=DMLDummyClassifier(), **kwargs)
    dml_did_ext.set_sample_splitting(all_smpls)
    np.random.seed(3141)
    dml_did_ext.fit(external_predictions=ext_predictions)

    res_dict = {"coef_normal": dml_did.coef[0], "coef_ext": dml_did_ext.coef[0]}

    return res_dict


@pytest.mark.ci
def test_doubleml_did_coef(doubleml_did_fixture):
    assert math.isclose(doubleml_did_fixture["coef_normal"], doubleml_did_fixture["coef_ext"], rel_tol=1e-9, abs_tol=1e-3)
