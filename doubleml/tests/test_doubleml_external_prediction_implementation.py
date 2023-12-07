import numpy as np
import pytest
from doubleml import DoubleMLCVAR, DoubleMLQTE, DoubleMLData
from doubleml.datasets import make_irm_data
from doubleml.utils import dummy_regressor, dummy_classifier

df_irm = make_irm_data(n_obs=500, dim_x=20, theta=0.5, return_type="DataFrame")

# CVAR
msg = "External predictions not implemented for DoubleMLCVAR."
ext_predictions = {"d": {}}
with pytest.raises(NotImplementedError, match=msg):
    cvar = DoubleMLCVAR(DoubleMLData(df_irm, "y", "d"), dummy_regressor(), dummy_classifier(), treatment=1)
    cvar.fit(external_predictions=ext_predictions)


# QTE
msg = "External predictions not implemented for DoubleMLQTE."
with pytest.raises(NotImplementedError, match=msg):
    qte = DoubleMLQTE(DoubleMLData(df_irm, "y", "d"), dummy_classifier(), dummy_classifier())
    cvar.fit(external_predictions=ext_predictions)
