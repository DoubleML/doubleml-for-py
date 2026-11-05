"""Refitting without storing predictions or models must not keep those of a previous fit."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import doubleml as dml
from doubleml.plm.datasets import make_plr_CCDDHNR2018


@pytest.mark.ci
def test_refit_without_storing_drops_previous_results():
    """A fit with store_predictions=False and store_models=False resets predictions, targets and models."""
    np.random.seed(3141)
    dml_data = make_plr_CCDDHNR2018(n_obs=200, dim_x=5)
    dml_plr = dml.DoubleMLPLR(dml_data, LinearRegression(), LinearRegression(), n_folds=2)
    dml_plr.fit(store_predictions=True, store_models=True)
    assert dml_plr.predictions is not None
    assert dml_plr.nuisance_targets is not None
    assert dml_plr.models is not None

    dml_plr.draw_sample_splitting()
    dml_plr.fit(store_predictions=False, store_models=False)
    assert dml_plr.predictions is None
    assert dml_plr.nuisance_targets is None
    assert dml_plr.models is None

    msg = r"predictions are None. Call .fit\(store_predictions=True\) to store the predictions."
    with pytest.raises(ValueError, match=msg):
        dml_plr.cate(basis=pd.DataFrame(np.ones((dml_data.n_obs, 1))))
    msg = r"Apply fit\(\) before evaluate_learners\(\)."
    with pytest.raises(ValueError, match=msg):
        dml_plr.evaluate_learners()
