import math

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml import DoubleMLData
from doubleml.plm.datasets import make_lplr_LZZ2020
from doubleml.plm.lplr import DoubleMLLPLR
from doubleml.utils import DMLDummyClassifier, DMLDummyRegressor


@pytest.fixture(scope="module", params=["nuisance_space", "instrument"])
def lplr_score(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def set_ml_m_ext(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def set_ml_t_ext(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def set_ml_M_ext(request):
    return request.param


@pytest.fixture(scope="module")
def doubleml_lplr_fixture(lplr_score, n_rep, set_ml_m_ext, set_ml_t_ext, set_ml_M_ext):
    ext_predictions = {"d": {}}

    x, y, d, _ = make_lplr_LZZ2020(n_obs=500, dim_x=20, alpha=0.5, return_type="np.array", treatment="continuous")

    np.random.seed(3141)
    dml_data = DoubleMLData.from_arrays(x=x, y=y, d=d)

    kwargs = {"obj_dml_data": dml_data, "score": lplr_score, "n_rep": n_rep}
    if lplr_score == "instrument":
        # ensure ml_a supports sample_weight
        kwargs["ml_a"] = LinearRegression()

    dml_lplr = DoubleMLLPLR(ml_M=LogisticRegression(max_iter=1000), ml_t=LinearRegression(), ml_m=LinearRegression(), **kwargs)
    np.random.seed(3141)
    dml_lplr.fit(store_predictions=True)

    # prepare external predictions and dummy learners
    if set_ml_M_ext:
        ext_predictions["d"]["ml_M"] = dml_lplr.predictions["ml_M"][:, :, 0]
        # provide inner predictions per inner fold index
        for i in range(dml_lplr.n_folds_inner):
            ext_predictions["d"][f"ml_M_inner_{i}"] = dml_lplr.predictions[f"ml_M_inner_{i}"][:, :, 0]
        ml_M = DMLDummyClassifier()
    else:
        ml_M = LogisticRegression(max_iter=1000)

    if set_ml_t_ext:
        ext_predictions["d"]["ml_t"] = dml_lplr.predictions["ml_t"][:, :, 0]
        ml_t = DMLDummyRegressor()
    else:
        ml_t = LinearRegression()

    if set_ml_m_ext:
        ext_predictions["d"]["ml_m"] = dml_lplr.predictions["ml_m"][:, :, 0]
        ml_m = DMLDummyRegressor()
        ext_predictions["d"]["ml_a"] = dml_lplr.predictions["ml_a"][:, :, 0]
        for i in range(dml_lplr.n_folds_inner):
            ext_predictions["d"][f"ml_a_inner_{i}"] = dml_lplr.predictions[f"ml_a_inner_{i}"][:, :, 0]
    else:
        ml_m = LinearRegression()

    # build second model with external predictions
    dml_lplr_ext = DoubleMLLPLR(ml_M=ml_M, ml_t=ml_t, ml_m=ml_m, **kwargs)

    np.random.seed(3141)
    dml_lplr_ext.fit(external_predictions=ext_predictions)

    res_dict = {
        "coef_normal": dml_lplr.coef[0],
        "coef_ext": dml_lplr_ext.coef[0],
        "se_normal": dml_lplr.se[0],
        "se_ext": dml_lplr_ext.se[0],
    }
    return res_dict


@pytest.mark.ci
def test_doubleml_lplr_coef(doubleml_lplr_fixture):
    assert math.isclose(doubleml_lplr_fixture["coef_normal"], doubleml_lplr_fixture["coef_ext"], rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_doubleml_lplr_se(doubleml_lplr_fixture):
    assert math.isclose(doubleml_lplr_fixture["se_normal"], doubleml_lplr_fixture["se_ext"], rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_doubleml_lplr_exceptions():
    ext_predictions = {"d": {}}

    x, y, d, _ = make_lplr_LZZ2020(n_obs=500, dim_x=20, alpha=0.5, return_type="np.array", treatment="continuous")

    np.random.seed(3141)
    dml_data = DoubleMLData.from_arrays(x=x, y=y, d=d)

    kwargs = {"obj_dml_data": dml_data}

    dml_lplr = DoubleMLLPLR(ml_M=LogisticRegression(max_iter=1000), ml_t=LinearRegression(), ml_m=LinearRegression(), **kwargs)
    np.random.seed(3141)
    dml_lplr.fit(store_predictions=True)

    # prepare external predictions and dummy learners

    ml_M = LogisticRegression(max_iter=1000)
    ml_t = LinearRegression()
    ml_m = LinearRegression()

    # build second model with external predictions
    dml_lplr_ext = DoubleMLLPLR(ml_M=ml_M, ml_t=ml_t, ml_m=ml_m, **kwargs)

    ext_predictions["d"]["ml_M"] = dml_lplr.predictions["ml_M"][:, :, 0]
    # provide inner predictions per inner fold index
    for i in range(dml_lplr.n_folds_inner - 1):
        ext_predictions["d"][f"ml_M_inner_{i}"] = dml_lplr.predictions[f"ml_M_inner_{i}"][:, :, 0]

    msg = r"When providing external predictions for ml_M, also inner predictions for all inner folds"
    with pytest.raises(ValueError, match=msg):
        dml_lplr_ext.fit(external_predictions=ext_predictions)

    ext_predictions["d"][f"ml_M_inner_{dml_lplr.n_folds_inner-1}"] = (dml_lplr.predictions)[
        f"ml_M_inner_{dml_lplr.n_folds_inner-1}"
    ][:, :, 0]
    ext_predictions["d"]["ml_a"] = dml_lplr.predictions["ml_a"][:, :, 0]
    for i in range(dml_lplr.n_folds_inner - 1):
        ext_predictions["d"][f"ml_a_inner_{i}"] = dml_lplr.predictions[f"ml_a_inner_{i}"][:, :, 0]

    msg = r"When providing external predictions for ml_a, also inner predictions for all inner folds"
    with pytest.raises(ValueError, match=msg):
        dml_lplr_ext.fit(external_predictions=ext_predictions)
