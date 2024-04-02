import numpy as np
import pytest
import math
from sklearn.linear_model import LinearRegression
from doubleml import DoubleMLPLIV, DoubleMLData
from doubleml.datasets import make_pliv_CHS2015
from doubleml.utils import DMLDummyRegressor


@pytest.fixture(scope="module", params=["partialling out", "IV-type"])
def score(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 3])
def dim_z(request):
    return request.param


@pytest.fixture(scope="module")
def adapted_doubleml_fixture(score, n_rep, dim_z):
    # IV-type score only allows dim_z = 1, so skip testcases with dim_z > 1 for IV-type score
    if dim_z > 1 and score == "IV-type":
        pytest.skip("IV-type score only allows dim_z = 1")
        res_dict = None
    else:
        ext_predictions = {"d": {}}

        data = make_pliv_CHS2015(
            n_obs=500, dim_x=20, alpha=0.5, dim_z=dim_z, return_type="DataFrame"
        )

        np.random.seed(3141)

        z_cols = [f"Z{i}" for i in range(1, dim_z + 1)]
        dml_data = DoubleMLData(data, "y", "d", z_cols=z_cols)

        kwargs = {
            "obj_dml_data": dml_data,
            "score": score,
            "n_rep": n_rep,
        }

        if score == "IV-type":
            kwargs["ml_g"] = LinearRegression()

        dml_pliv = DoubleMLPLIV(
            ml_m=LinearRegression(),
            ml_l=LinearRegression(),
            ml_r=LinearRegression(),
            **kwargs,
        )
        np.random.seed(3141)

        dml_pliv.fit(store_predictions=True)

        ext_predictions["d"]["ml_l"] = dml_pliv.predictions["ml_l"][:, :, 0]
        ext_predictions["d"]["ml_r"] = dml_pliv.predictions["ml_r"][:, :, 0]

        if dim_z == 1:
            ext_predictions["d"]["ml_m"] = dml_pliv.predictions["ml_m"][:, :, 0]
            if score == "IV-type":
                kwargs["ml_g"] = DMLDummyRegressor()
                ext_predictions["d"]["ml_g"] = dml_pliv.predictions["ml_g"][:, :, 0]
        else:
            for instr in range(dim_z):
                ml_m_key = "ml_m_" + "Z" + str(instr + 1)
                ext_predictions["d"][ml_m_key] = dml_pliv.predictions[ml_m_key][:, :, 0]

        dml_pliv_ext = DoubleMLPLIV(
            ml_m=DMLDummyRegressor(), ml_l=DMLDummyRegressor(), ml_r=DMLDummyRegressor(), **kwargs
        )

        np.random.seed(3141)
        dml_pliv_ext.fit(external_predictions=ext_predictions)

        res_dict = {"coef_normal": dml_pliv.coef[0], "coef_ext": dml_pliv_ext.coef[0]}

    return res_dict


@pytest.mark.ci
def test_adapted_doubleml_coef(adapted_doubleml_fixture):
    assert math.isclose(
        adapted_doubleml_fixture["coef_normal"],
        adapted_doubleml_fixture["coef_ext"],
        rel_tol=1e-9,
        abs_tol=1e-4,
    )
