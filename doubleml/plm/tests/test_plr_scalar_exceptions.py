import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso

import doubleml as dml
from doubleml.plm.datasets import make_plr_CCDDHNR2018
from doubleml.plm.plr_scalar import PLR

np.random.seed(3141)
obj_dml_data = make_plr_CCDDHNR2018(n_obs=100, dim_x=10, alpha=0.5)

# Create data with instruments for IV check
df = obj_dml_data.data.copy()
x_cols = [c for c in df.columns if c.startswith("X")]
dml_data_iv = dml.DoubleMLData(df, y_col="y", d_cols="d", x_cols=x_cols[:-1], z_cols=x_cols[-1])

ml_l = Lasso(alpha=0.1)
ml_m = Lasso(alpha=0.1)
ml_g = Lasso(alpha=0.1)


@pytest.mark.ci
def test_plr_scalar_exception_data():
    msg = r"The data must be of DoubleMLData type\."
    with pytest.raises(TypeError, match=msg):
        PLR(pd.DataFrame(), ml_l, ml_m)


@pytest.mark.ci
def test_plr_scalar_exception_instrument():
    msg = r"Incompatible data\. .* have been set as instrumental variable\(s\)\."
    with pytest.raises(ValueError, match=msg):
        PLR(dml_data_iv, ml_l, ml_m)


@pytest.mark.ci
def test_plr_scalar_exception_score():
    msg = r"Invalid score 'invalid'\."
    with pytest.raises(ValueError, match=msg):
        PLR(obj_dml_data, ml_l, ml_m, score="invalid")


@pytest.mark.ci
def test_plr_scalar_exception_n_folds():
    dml_obj = PLR(obj_dml_data, ml_l, ml_m)
    msg = r"n_folds must be an integer >= 2\."
    with pytest.raises(ValueError, match=msg):
        dml_obj.draw_sample_splitting(n_folds=1)
    with pytest.raises(ValueError, match=msg):
        dml_obj.draw_sample_splitting(n_folds=0)


@pytest.mark.ci
def test_plr_scalar_exception_n_rep():
    dml_obj = PLR(obj_dml_data, ml_l, ml_m)
    msg = r"n_rep must be an integer >= 1\."
    with pytest.raises(ValueError, match=msg):
        dml_obj.draw_sample_splitting(n_rep=0)


@pytest.mark.ci
def test_plr_scalar_exception_fit_nuisance_without_smpls():
    dml_obj = PLR(obj_dml_data, ml_l, ml_m)
    msg = r"Sample splitting has not been initialized\."
    with pytest.raises(ValueError, match=msg):
        dml_obj.fit_nuisance_models()


@pytest.mark.ci
def test_plr_scalar_exception_estimate_causal_without_predictions():
    dml_obj = PLR(obj_dml_data, ml_l, ml_m)
    dml_obj.draw_sample_splitting()
    msg = r"Predictions not available\."
    with pytest.raises(ValueError, match=msg):
        dml_obj.estimate_causal_parameters()


@pytest.mark.ci
def test_plr_scalar_warning_ml_g_partialling_out():
    with pytest.warns(UserWarning, match="will be ignored"):
        PLR(obj_dml_data, ml_l, ml_m, ml_g, score="partialling out")


@pytest.mark.ci
def test_plr_scalar_warning_ml_g_iv_type_missing():
    with pytest.warns(UserWarning, match="ml_l and ml_g should be specified"):
        PLR(obj_dml_data, ml_l, ml_m, score="IV-type")
