import numpy as np
import pytest
import math
from sklearn.linear_model import LinearRegression, LogisticRegression
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


@pytest.fixture(scope="module", params=[True, False])
def set_ml_m_ext(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def set_ml_g_ext(request):
    return request.param


@pytest.fixture(scope="module")
def doubleml_irm_fixture(irm_score, dml_procedure, n_rep, set_ml_m_ext, set_ml_g_ext):
    ext_predictions = {"d": {}}

    x, y, d = make_irm_data(n_obs=500, dim_x=20, theta=0.5, return_type="np.array")

    np.random.seed(3141)

    dml_data = DoubleMLData.from_arrays(x=x, y=y, d=d)

    kwargs = {"obj_dml_data": dml_data, "score": irm_score, "n_rep": n_rep, "dml_procedure": dml_procedure}

    DMLIRM = DoubleMLIRM(ml_g=LinearRegression(), ml_m=LogisticRegression(), **kwargs)
    np.random.seed(3141)

    DMLIRM.fit(store_predictions=True)

    if set_ml_m_ext:
        ext_predictions["d"]["ml_m"] = DMLIRM.predictions["ml_m"][:, :, 0]
        ml_m = dummy_classifier()
    else:
        ml_m = LogisticRegression(random_state=42)

    if set_ml_g_ext:
        ext_predictions["d"]["ml_g0"] = DMLIRM.predictions["ml_g0"][:, :, 0]
        ext_predictions["d"]["ml_g1"] = DMLIRM.predictions["ml_g1"][:, :, 0]
        ml_g = dummy_regressor()
    else:
        ml_g = LinearRegression()

    DMLIRM_ext = DoubleMLIRM(ml_g=ml_g, ml_m=ml_m, **kwargs)

    np.random.seed(3141)
    DMLIRM_ext.fit(external_predictions=ext_predictions)

    res_dict = {"coef_normal": DMLIRM.coef[0], "coef_ext": DMLIRM_ext.coef[0]}

    return res_dict


@pytest.mark.ci
def test_doubleml_irm_coef(doubleml_irm_fixture):
    assert math.isclose(doubleml_irm_fixture["coef_normal"], doubleml_irm_fixture["coef_ext"], rel_tol=1e-9, abs_tol=1e-4)
