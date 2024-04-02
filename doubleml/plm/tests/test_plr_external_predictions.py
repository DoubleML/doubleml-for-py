import numpy as np
import pytest
import math
from sklearn.linear_model import LinearRegression
from doubleml import DoubleMLPLR, DoubleMLData
from doubleml.datasets import make_plr_CCDDHNR2018
from doubleml.utils import DMLDummyRegressor


@pytest.fixture(scope="module", params=["IV-type", "partialling out"])
def plr_score(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def set_ml_m_ext(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def set_ml_l_ext(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def set_ml_g_ext(request):
    return request.param


@pytest.fixture(scope="module")
def doubleml_plr_fixture(plr_score, n_rep, set_ml_m_ext, set_ml_l_ext, set_ml_g_ext):
    ext_predictions = {"d": {}}

    x, y, d = make_plr_CCDDHNR2018(n_obs=500, dim_x=20, alpha=0.5, return_type="np.array")

    np.random.seed(3141)

    dml_data = DoubleMLData.from_arrays(x=x, y=y, d=d)

    kwargs = {"obj_dml_data": dml_data, "score": plr_score, "n_rep": n_rep}

    if plr_score == "IV-type":
        kwargs["ml_g"] = LinearRegression()

    dml_plr = DoubleMLPLR(ml_m=LinearRegression(), ml_l=LinearRegression(), **kwargs)
    np.random.seed(3141)

    dml_plr.fit(store_predictions=True)

    if set_ml_m_ext:
        ext_predictions["d"]["ml_m"] = dml_plr.predictions["ml_m"][:, :, 0]
        ml_m = DMLDummyRegressor()
    else:
        ml_m = LinearRegression()

    if set_ml_l_ext:
        ext_predictions["d"]["ml_l"] = dml_plr.predictions["ml_l"][:, :, 0]
        ml_l = DMLDummyRegressor()
    else:
        ml_l = LinearRegression()

    if plr_score == "IV-type" and set_ml_g_ext:
        ext_predictions["d"]["ml_g"] = dml_plr.predictions["ml_g"][:, :, 0]
        kwargs["ml_g"] = DMLDummyRegressor()
    elif plr_score == "IV-type" and not set_ml_g_ext:
        kwargs["ml_g"] = LinearRegression()
    else:
        pass

    if plr_score == "IV-type" and set_ml_g_ext and not set_ml_l_ext:
        ml_l = DMLDummyRegressor()

    # special case if ml_l is not needed
    dml_plr_ext = DoubleMLPLR(ml_m=ml_m, ml_l=ml_l, **kwargs)

    np.random.seed(3141)
    dml_plr_ext.fit(external_predictions=ext_predictions)

    res_dict = {"coef_normal": dml_plr.coef[0], "coef_ext": dml_plr_ext.coef[0]}

    return res_dict


@pytest.mark.ci
def test_doubleml_plr_coef(doubleml_plr_fixture):
    assert math.isclose(doubleml_plr_fixture["coef_normal"], doubleml_plr_fixture["coef_ext"], rel_tol=1e-9, abs_tol=1e-4)
