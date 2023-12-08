import pytest
from doubleml import DoubleMLCVAR, DoubleMLQTE, DoubleMLData
from doubleml.datasets import make_irm_data
from doubleml.utils import dummy_regressor, dummy_classifier

df_irm = make_irm_data(n_obs=10, dim_x=2, theta=0.5, return_type="DataFrame")
ext_predictions = {"d": {}}


@pytest.mark.ci
def test_cvar_external_prediction_exception():
    msg = "External predictions not implemented for DoubleMLCVAR."
    with pytest.raises(NotImplementedError, match=msg):
        cvar = DoubleMLCVAR(DoubleMLData(df_irm, "y", "d"), dummy_regressor(), dummy_classifier(), treatment=1)
        cvar.fit(external_predictions=ext_predictions)


@pytest.mark.ci
def test_qte_external_prediction_exception():
    msg = "External predictions not implemented for DoubleMLQTE."
    with pytest.raises(NotImplementedError, match=msg):
        qte = DoubleMLQTE(DoubleMLData(df_irm, "y", "d"), dummy_classifier(), dummy_classifier())
        qte.fit(external_predictions=ext_predictions)
