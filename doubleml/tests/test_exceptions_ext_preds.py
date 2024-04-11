import pytest
from doubleml import DoubleMLCVAR, DoubleMLQTE, DoubleMLIRM, DoubleMLData
from doubleml.datasets import make_irm_data
from doubleml.utils import DMLDummyRegressor, DMLDummyClassifier

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

df_irm = make_irm_data(n_obs=10, dim_x=2, theta=0.5, return_type="DataFrame")
ext_predictions = {"d": {}}


@pytest.mark.ci
def test_cvar_external_prediction_exception():
    msg = "External predictions not implemented for DoubleMLCVAR."
    with pytest.raises(NotImplementedError, match=msg):
        cvar = DoubleMLCVAR(DoubleMLData(df_irm, "y", "d"), DMLDummyRegressor(), DMLDummyClassifier(), treatment=1)
        cvar.fit(external_predictions=ext_predictions)


@pytest.mark.ci
def test_qte_external_prediction_exception():
    msg = "External predictions not implemented for DoubleMLQTE."
    with pytest.raises(NotImplementedError, match=msg):
        qte = DoubleMLQTE(DoubleMLData(df_irm, "y", "d"), DMLDummyClassifier(), DMLDummyClassifier())
        qte.fit(external_predictions=ext_predictions)


@pytest.mark.ci
def test_sensitivity_benchmark_external_prediction_exception():
    msg = "fit_args must be a dict. "
    with pytest.raises(TypeError, match=msg):
        fit_args = []
        irm = DoubleMLIRM(DoubleMLData(df_irm, "y", "d"), RandomForestRegressor(), RandomForestClassifier())
        irm.fit()
        irm.sensitivity_analysis()
        irm.sensitivity_benchmark(benchmarking_set=["X1"], fit_args=fit_args)
