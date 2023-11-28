import numpy as np
import pytest
import math
from sklearn.linear_model import LinearRegression, LassoCV, LogisticRegression
from doubleml import DoubleMLIRM, DoubleMLData
from doubleml.datasets import make_irm_data
from doubleml.utils import dummy_regressor, dummy_classifier


@pytest.fixture(scope="module", params=["ATE", "ATTE"])
def irm_score(request):
    return request.param


@pytest.fixture(scope="module", params=["dml1", "dml2"])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module")
def doubleml_irm_fixture(irm_score, dml_procedure, n_rep):
    ext_predictions = {"d": {}}

    x, y, d = make_irm_data(n_obs=500, dim_x=20, theta=0.5, return_type="np.array")

    np.random.seed(3141)

    dml_data = DoubleMLData.from_arrays(x=x, y=y, d=d)

    kwargs = {"obj_dml_data": dml_data, "score": irm_score, "n_rep": n_rep, "dml_procedure": dml_procedure}

    DMLIRM = DoubleMLIRM(ml_g=LinearRegression(), ml_m=LogisticRegression(), **kwargs)
    np.random.seed(3141)

    DMLIRM.fit(store_predictions=True)

    ext_predictions["d"]["ml_g0"] = DMLIRM.predictions["ml_g0"][:, :, 0]
    ext_predictions["d"]["ml_g1"] = DMLIRM.predictions["ml_g1"][:, :, 0]
    ext_predictions["d"]["ml_m"] = DMLIRM.predictions["ml_m"][:, :, 0]

    DMLIRM_ext = DoubleMLIRM(ml_g=dummy_regressor(), ml_m=dummy_classifier(), **kwargs)

    np.random.seed(3141)
    DMLIRM_ext.fit(external_predictions=ext_predictions)

    res_dict = {"coef_normal": DMLIRM.coef, "coef_ext": DMLIRM_ext.coef}

    return res_dict


@pytest.mark.ci
def test_doubleml_plr_coef(doubleml_plr_fixture):
    assert math.isclose(doubleml_plr_fixture["coef_normal"], doubleml_plr_fixture["coef_ext"], rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_doubleml_irm_coef(doubleml_irm_fixture):
    assert math.isclose(doubleml_irm_fixture["coef_normal"], doubleml_irm_fixture["coef_ext"], rel_tol=1e-9, abs_tol=1e-4)
