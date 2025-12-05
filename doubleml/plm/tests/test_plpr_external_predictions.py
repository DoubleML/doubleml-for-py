import math

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

import doubleml as dml
from doubleml.plm.datasets import make_plpr_CP2025
from doubleml.utils import DMLDummyRegressor

treat_label = {"cre_general": "d", "cre_normal": "d", "fd_exact": "d_diff", "wg_approx": "d_demean"}


@pytest.fixture(scope="module", params=["IV-type", "partialling out"])
def plpr_score(request):
    return request.param


@pytest.fixture(scope="module", params=["cre_general", "cre_normal", "fd_exact", "wg_approx"])
def plpr_approach(request):
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
def doubleml_plpr_fixture(plpr_score, plpr_approach, n_rep, set_ml_m_ext, set_ml_l_ext, set_ml_g_ext):
    ext_predictions = {treat_label[plpr_approach]: {}}

    plpr_data = make_plpr_CP2025(num_id=100, theta=0.5, dgp_type="dgp1")

    np.random.seed(3141)
    dml_data_plpr = dml.DoubleMLPanelData(
        plpr_data,
        y_col="y",
        d_cols="d",
        t_col="time",
        id_col="id",
        static_panel=True,
    )

    kwargs = {"obj_dml_data": dml_data_plpr, "score": plpr_score, "approach": plpr_approach, "n_rep": n_rep}

    if plpr_score == "IV-type":
        kwargs["ml_g"] = LinearRegression()

    dml_plpr = dml.DoubleMLPLPR(ml_m=LinearRegression(), ml_l=LinearRegression(), **kwargs)

    np.random.seed(3141)
    dml_plpr.fit(store_predictions=True)

    if set_ml_m_ext:
        ext_predictions[treat_label[plpr_approach]]["ml_m"] = dml_plpr.predictions["ml_m"][:, :, 0]
        ml_m = DMLDummyRegressor()
    else:
        ml_m = LinearRegression()

    if set_ml_l_ext:
        ext_predictions[treat_label[plpr_approach]]["ml_l"] = dml_plpr.predictions["ml_l"][:, :, 0]
        ml_l = DMLDummyRegressor()
    else:
        ml_l = LinearRegression()

    if plpr_score == "IV-type" and set_ml_g_ext:
        ext_predictions[treat_label[plpr_approach]]["ml_g"] = dml_plpr.predictions["ml_g"][:, :, 0]
        kwargs["ml_g"] = DMLDummyRegressor()
    elif plpr_score == "IV-type" and not set_ml_g_ext:
        kwargs["ml_g"] = LinearRegression()
    else:
        pass

    if plpr_score == "IV-type" and set_ml_g_ext and not set_ml_l_ext:
        ml_l = DMLDummyRegressor()

    # special case if ml_l is not needed
    dml_plpr_ext = dml.DoubleMLPLPR(ml_m=ml_m, ml_l=ml_l, **kwargs)

    np.random.seed(3141)
    dml_plpr_ext.fit(external_predictions=ext_predictions)

    res_dict = {"coef_normal": dml_plpr.coef[0], "coef_ext": dml_plpr_ext.coef[0]}

    return res_dict


@pytest.mark.ci
def test_doubleml_plpr_coef(doubleml_plpr_fixture):
    assert math.isclose(doubleml_plpr_fixture["coef_normal"], doubleml_plpr_fixture["coef_ext"], rel_tol=1e-9, abs_tol=1e-4)
