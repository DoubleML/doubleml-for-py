import numpy as np
import pytest
import math
from sklearn.linear_model import LogisticRegression, LinearRegression
from doubleml import DoubleMLCVAR, DoubleMLData
from doubleml.datasets import make_irm_data
from doubleml.utils import dummy_regressor, dummy_classifier
from ._utils import draw_smpls


@pytest.fixture(scope="module", params=["dml1", "dml2"])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def normalize_ipw(request):
    return request.param


@pytest.fixture(scope="module")
def doubleml_cvar_fixture(dml_procedure, n_rep, normalize_ipw):
    ext_predictions = {"d": {}}
    np.random.seed(3141)
    data = make_irm_data(theta=0.5, n_obs=500, dim_x=20, return_type="DataFrame")

    dml_data = DoubleMLData(data, "y", "d")
    all_smpls = draw_smpls(len(dml_data.y), 5, n_rep=n_rep, groups=dml_data.d)

    kwargs = {
        "obj_dml_data": dml_data,
        "score": "CVaR",
        "n_rep": n_rep,
        "dml_procedure": dml_procedure,
        "normalize_ipw": normalize_ipw,
        "draw_sample_splitting": False,
        "treatment": 1, 
        "quantile": 0.5
    }

    ml_g = LinearRegression()
    ml_m = LogisticRegression()

    DMLcvar = DoubleMLCVAR(ml_g=ml_g, ml_m=ml_m, **kwargs)
    DMLcvar.set_sample_splitting(all_smpls)
    np.random.seed(3141)

    DMLcvar.fit(store_predictions=True)

    ext_predictions["d"]["ml_g"] = DMLcvar.predictions["ml_g"][:, :, 0]
    ext_predictions["d"]["ml_m"] = DMLcvar.predictions["ml_m"][:, :, 0]

    DMLcvar_ext = DoubleMLCVAR(ml_g=dummy_regressor(), ml_m=dummy_classifier(), **kwargs)
    DMLcvar_ext.set_sample_splitting(all_smpls)

    np.random.seed(3141)
    DMLcvar_ext.fit(external_predictions=ext_predictions)

    res_dict = {"coef_normal": DMLcvar.coef, "coef_ext": DMLcvar_ext.coef}

    return res_dict


@pytest.mark.ci
def test_doubleml_cvar_coef(doubleml_cvar_fixture):
    assert math.isclose(doubleml_cvar_fixture["coef_normal"], doubleml_cvar_fixture["coef_ext"], rel_tol=1e-9, abs_tol=1e-4)
