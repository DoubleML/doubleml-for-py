import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso, LogisticRegression

import doubleml as dml
from doubleml.plm.datasets import make_plr_CCDDHNR2018
from doubleml.plm.plr_scalar import PLR

np.random.seed(3141)
obj_dml_data = make_plr_CCDDHNR2018(n_obs=100, dim_x=10, alpha=0.5)

# Binary-outcome data for binary-specific tests
np.random.seed(42)
_n = 100
_X = np.random.normal(size=(_n, 3))
_d = (np.random.normal(size=_n) > 0).astype(float)
_y_bin = (np.random.normal(size=_n) > 0).astype(float)
_df_binary = pd.DataFrame({"y": _y_bin, "d": _d, "X1": _X[:, 0], "X2": _X[:, 1], "X3": _X[:, 2]})
obj_dml_data_binary = dml.DoubleMLData(_df_binary, y_col="y", d_cols="d", x_cols=["X1", "X2", "X3"])

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
        PLR(pd.DataFrame())


@pytest.mark.ci
def test_plr_scalar_exception_instrument():
    msg = r"Incompatible data\. .* have been set as instrumental variable\(s\)\."
    with pytest.raises(ValueError, match=msg):
        PLR(dml_data_iv)


@pytest.mark.ci
def test_plr_scalar_exception_score():
    msg = r"Invalid score 'invalid'\."
    with pytest.raises(ValueError, match=msg):
        PLR(obj_dml_data, score="invalid")


@pytest.mark.ci
def test_plr_scalar_exception_n_folds():
    dml_obj = PLR(obj_dml_data)
    msg = r"n_folds must be an integer >= 2\."
    with pytest.raises(ValueError, match=msg):
        dml_obj.draw_sample_splitting(n_folds=1)
    with pytest.raises(ValueError, match=msg):
        dml_obj.draw_sample_splitting(n_folds=0)


@pytest.mark.ci
def test_plr_scalar_exception_n_rep():
    dml_obj = PLR(obj_dml_data)
    msg = r"n_rep must be an integer >= 1\."
    with pytest.raises(ValueError, match=msg):
        dml_obj.draw_sample_splitting(n_rep=0)


@pytest.mark.ci
def test_plr_scalar_exception_fit_nuisance_without_smpls():
    dml_obj = PLR(obj_dml_data)
    dml_obj.set_learners(ml_l=ml_l, ml_m=ml_m)
    msg = r"Sample splitting has not been initialized\."
    with pytest.raises(ValueError, match=msg):
        dml_obj.fit_nuisance_models()


@pytest.mark.ci
def test_plr_scalar_exception_estimate_causal_without_predictions():
    dml_obj = PLR(obj_dml_data)
    dml_obj.set_learners(ml_l=ml_l, ml_m=ml_m)
    dml_obj.draw_sample_splitting()
    msg = r"Predictions not available\."
    with pytest.raises(ValueError, match=msg):
        dml_obj.estimate_causal_parameters()


@pytest.mark.ci
def test_plr_scalar_warning_ml_g_partialling_out():
    dml_obj = PLR(obj_dml_data, score="partialling out")
    with pytest.warns(UserWarning, match="not required for score.*ignored"):
        dml_obj.set_learners(ml_l=ml_l, ml_m=ml_m, ml_g=ml_g)


@pytest.mark.ci
def test_plr_scalar_exception_missing_learner():
    dml_obj = PLR(obj_dml_data)
    dml_obj.draw_sample_splitting()
    msg = r"Learner 'ml_l' is required but not set"
    with pytest.raises(ValueError, match=msg):
        dml_obj.fit()


@pytest.mark.ci
def test_plr_scalar_exception_missing_learner_partial():
    dml_obj = PLR(obj_dml_data)
    dml_obj.set_learners(ml_l=ml_l)
    dml_obj.draw_sample_splitting()
    msg = r"Learner 'ml_m' is required but not set"
    with pytest.raises(ValueError, match=msg):
        dml_obj.fit()


@pytest.mark.ci
def test_plr_scalar_exception_invalid_learner():
    dml_obj = PLR(obj_dml_data)
    msg = r"Invalid learner provided for ml_l: provide an instance"
    with pytest.raises(TypeError, match=msg):
        dml_obj.set_learners(ml_l=Lasso)  # class instead of instance


@pytest.mark.ci
def test_plr_scalar_exception_iv_type_binary_outcome():
    """IV-type score with binary outcome raises ValueError."""
    msg = r"For score = 'IV-type', additive probability models \(binary outcomes\) are not supported\."
    with pytest.raises(ValueError, match=msg):
        PLR(obj_dml_data_binary, score="IV-type")


@pytest.mark.ci
def test_plr_scalar_warning_binary_outcome_classifier():
    """Classifier ml_l with binary outcome warns about fitting an additive probability model."""
    dml_obj = PLR(obj_dml_data_binary)
    msg = r"The ml_l learner .+ was identified as classifier\. Fitting an additive probability model\."
    with pytest.warns(UserWarning, match=msg):
        dml_obj.set_learners(ml_l=LogisticRegression(), ml_m=Lasso())
