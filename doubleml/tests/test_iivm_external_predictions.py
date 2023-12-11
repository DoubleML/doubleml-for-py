import numpy as np
import pytest
import math
from sklearn.linear_model import LinearRegression, LogisticRegression
from doubleml import DoubleMLIIVM, DoubleMLData
from doubleml.datasets import make_iivm_data
from doubleml.utils import dummy_regressor, dummy_classifier


@pytest.fixture(scope="module", params=["dml1", "dml2"])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module")
def adapted_doubleml_fixture(dml_procedure, n_rep):
    ext_predictions = {"d": {}}

    data = make_iivm_data(
        n_obs=500, dim_x=20, theta=0.5, alpha_x=1.0, return_type="DataFrame"
    )

    np.random.seed(3141)

    dml_data = DoubleMLData(data, "y", "d", z_cols="z")

    kwargs = {
        "obj_dml_data": dml_data,
        "score": "LATE",
        "n_rep": n_rep,
        "dml_procedure": dml_procedure,
    }

    dml_iivm = DoubleMLIIVM(
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(),
        ml_r=LogisticRegression(),
        **kwargs,
    )
    np.random.seed(3141)

    dml_iivm.fit(store_predictions=True)

    ext_predictions["d"]["ml_g0"] = dml_iivm.predictions["ml_g0"][:, :, 0]
    ext_predictions["d"]["ml_g1"] = dml_iivm.predictions["ml_g1"][:, :, 0]
    ext_predictions["d"]["ml_m"] = dml_iivm.predictions["ml_m"][:, :, 0]
    ext_predictions["d"]["ml_r0"] = dml_iivm.predictions["ml_r0"][:, :, 0]
    ext_predictions["d"]["ml_r1"] = dml_iivm.predictions["ml_r1"][:, :, 0]

    dml_iivm_ext = DoubleMLIIVM(
        ml_g=dummy_regressor(), ml_m=dummy_classifier(), ml_r=dummy_classifier(), **kwargs
    )

    np.random.seed(3141)
    dml_iivm_ext.fit(external_predictions=ext_predictions)

    res_dict = {"coef_normal": dml_iivm.coef, "coef_ext": dml_iivm_ext.coef}

    return res_dict


@pytest.mark.ci
def test_adapted_doubleml_coef(adapted_doubleml_fixture):
    assert math.isclose(
        adapted_doubleml_fixture["coef_normal"],
        adapted_doubleml_fixture["coef_ext"],
        rel_tol=1e-9,
        abs_tol=1e-4,
    )
