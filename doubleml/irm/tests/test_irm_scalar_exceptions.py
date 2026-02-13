import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression

import doubleml as dml
from doubleml.irm.datasets import make_irm_data
from doubleml.irm.irm_scalar import IRM
from doubleml.plm.datasets import make_plr_CCDDHNR2018

np.random.seed(3141)
obj_dml_data = make_irm_data(theta=0.5, n_obs=100, dim_x=10, return_type="DoubleMLData")

# Binary-outcome data for binary predictions check tests
np.random.seed(42)
_n = 200
_X = np.random.normal(size=(_n, 3))
_d_bin = (np.random.normal(size=_n) > 0).astype(float)
_y_bin = (np.random.normal(size=_n) > 0).astype(float)
_df_binary = pd.DataFrame({"y": _y_bin, "d": _d_bin, "X1": _X[:, 0], "X2": _X[:, 1], "X3": _X[:, 2]})
obj_dml_data_binary = dml.DoubleMLData(_df_binary, y_col="y", d_cols="d", x_cols=["X1", "X2", "X3"])


class _HardLabelClassifier(RandomForestClassifier):
    """Classifier that returns hard 0/1 labels instead of probabilities — for testing only."""

    def predict_proba(self, X):
        preds = np.zeros((len(X), 2))
        preds[:, 1] = (np.arange(len(X)) % 2).astype(float)
        preds[:, 0] = 1.0 - preds[:, 1]
        return preds


ml_g = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
ml_m = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)


@pytest.mark.ci
def test_irm_scalar_exception_data():
    msg = r"The data must be of DoubleMLData type\."
    with pytest.raises(TypeError, match=msg):
        IRM(pd.DataFrame())


@pytest.mark.ci
def test_irm_scalar_exception_instrument():
    # Create data with instruments
    np.random.seed(3141)
    plr_data = make_plr_CCDDHNR2018(n_obs=100, dim_x=10, alpha=0.5)
    df = plr_data.data.copy()
    x_cols = [c for c in df.columns if c.startswith("X")]

    import doubleml as dml

    dml_data_iv = dml.DoubleMLData(df, y_col="y", d_cols="d", x_cols=x_cols[:-1], z_cols=x_cols[-1])

    msg = r"Incompatible data\. .* have been set as instrumental variable\(s\)\."
    with pytest.raises(ValueError, match=msg):
        IRM(dml_data_iv)


@pytest.mark.ci
def test_irm_scalar_exception_non_binary_treatment():
    # Create data with continuous treatment
    np.random.seed(3141)
    plr_data = make_plr_CCDDHNR2018(n_obs=100, dim_x=10, alpha=0.5)
    msg = r"Incompatible data.*exactly one binary variable"
    with pytest.raises(ValueError, match=msg):
        IRM(plr_data)


@pytest.mark.ci
def test_irm_scalar_exception_score():
    msg = r"Invalid score"
    with pytest.raises(ValueError, match=msg):
        IRM(obj_dml_data, score="invalid")


@pytest.mark.ci
def test_irm_scalar_exception_n_folds():
    dml_obj = IRM(obj_dml_data)
    msg = r"n_folds must be an integer >= 2\."
    with pytest.raises(ValueError, match=msg):
        dml_obj.draw_sample_splitting(n_folds=1)
    with pytest.raises(ValueError, match=msg):
        dml_obj.draw_sample_splitting(n_folds=0)


@pytest.mark.ci
def test_irm_scalar_exception_n_rep():
    dml_obj = IRM(obj_dml_data)
    msg = r"n_rep must be an integer >= 1\."
    with pytest.raises(ValueError, match=msg):
        dml_obj.draw_sample_splitting(n_rep=0)


@pytest.mark.ci
def test_irm_scalar_exception_fit_nuisance_without_smpls():
    dml_obj = IRM(obj_dml_data, ml_g=ml_g, ml_m=ml_m)
    msg = r"Sample splitting has not been initialized\."
    with pytest.raises(ValueError, match=msg):
        dml_obj.fit_nuisance_models()


@pytest.mark.ci
def test_irm_scalar_exception_estimate_causal_without_predictions():
    dml_obj = IRM(obj_dml_data, ml_g=ml_g, ml_m=ml_m)
    dml_obj.draw_sample_splitting()
    msg = r"Predictions not available\."
    with pytest.raises(ValueError, match=msg):
        dml_obj.estimate_causal_parameters()


@pytest.mark.ci
def test_irm_scalar_exception_missing_learner():
    dml_obj = IRM(obj_dml_data)
    dml_obj.draw_sample_splitting()
    msg = r"Learner 'ml_g0' is required but not set"
    with pytest.raises(ValueError, match=msg):
        dml_obj.fit()


@pytest.mark.ci
def test_irm_scalar_exception_missing_learner_partial():
    dml_obj = IRM(obj_dml_data)
    dml_obj.set_learners(ml_g=ml_g)
    dml_obj.draw_sample_splitting()
    msg = r"Learner 'ml_m' is required but not set"
    with pytest.raises(ValueError, match=msg):
        dml_obj.fit()


@pytest.mark.ci
def test_irm_scalar_exception_invalid_learner():
    dml_obj = IRM(obj_dml_data)
    msg = r"Invalid learner provided for ml_g: provide an instance"
    with pytest.raises(TypeError, match=msg):
        dml_obj.set_learners(ml_g=RandomForestRegressor)  # class instead of instance


@pytest.mark.ci
def test_irm_scalar_exception_ml_m_regressor():
    dml_obj = IRM(obj_dml_data)
    # LinearRegression is a regressor, not allowed for ml_m; warns then raises TypeError (no predict_proba)
    with pytest.raises(TypeError, match=r"has no method .predict_proba"):
        dml_obj.set_learners(ml_m=LinearRegression())


@pytest.mark.ci
def test_irm_scalar_exception_normalize_ipw_type():
    msg = r"Normalization indicator has to be boolean"
    with pytest.raises(TypeError, match=msg):
        IRM(obj_dml_data, normalize_ipw="True")


@pytest.mark.ci
def test_irm_scalar_exception_binary_predictions_g():
    """Classifier ml_g returning hard labels (0/1) instead of probabilities raises ValueError."""
    ml_m_test = RandomForestClassifier(n_estimators=5, random_state=42)
    dml_obj = IRM(obj_dml_data_binary, ml_g=_HardLabelClassifier(), ml_m=ml_m_test)
    dml_obj.draw_sample_splitting(n_folds=3)
    msg = r"For the binary variable .+, predictions .+ are also observed to be binary"
    with pytest.raises(ValueError, match=msg):
        dml_obj.fit_nuisance_models()
