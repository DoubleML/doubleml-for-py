import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml.data.did_data import DoubleMLDIDData
from doubleml.did.did import DoubleMLDID
from doubleml.did.did_cs import DoubleMLDIDCS


@pytest.mark.ci
def test_deprecation_DoubleMLDIDData(generate_data_did):
    (x, y, d, _) = generate_data_did
    with pytest.warns(FutureWarning, match="DoubleMLDIDData is deprecated"):
        _ = DoubleMLDIDData.from_arrays(x, y, d)


@pytest.mark.ci
def test_deprecation_DoubleMLDID(generate_data_did):
    (x, y, d, _) = generate_data_did
    obj_dml_data = DoubleMLDIDData.from_arrays(x, y, d)
    with pytest.warns(DeprecationWarning, match="DoubleMLDID is deprecated"):
        _ = DoubleMLDID(obj_dml_data, ml_g=LinearRegression(), ml_m=LogisticRegression())


@pytest.mark.ci
def test_deprecation_DoubleMLDIDCS(generate_data_did_cs):
    (x, y, d, t) = generate_data_did_cs
    obj_dml_data = DoubleMLDIDData.from_arrays(x, y, d, t=t)
    with pytest.warns(DeprecationWarning, match="DoubleMLDIDCS is deprecated"):
        _ = DoubleMLDIDCS(obj_dml_data, ml_g=LinearRegression(), ml_m=LogisticRegression())
