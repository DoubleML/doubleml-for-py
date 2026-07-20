"""Test external prediction validation for scalar DoubleML models."""

import numpy as np
import pytest
from sklearn.linear_model import Lasso

from doubleml.plm.datasets import make_plr_CCDDHNR2018
from doubleml.plm.plr_scalar import PLR


@pytest.mark.ci
def test_scalar_external_predictions_unknown_key():
    """Reject external predictions with unknown learner keys."""
    np.random.seed(3141)
    dml_data = make_plr_CCDDHNR2018(n_obs=10)
    dml_obj = PLR(dml_data)
    dml_obj.set_learners(ml_l=Lasso(), ml_m=Lasso())
    dml_obj.draw_sample_splitting(n_folds=2, n_rep=1)

    ext_predictions = {
        "ml_l": np.zeros((10, 1)),
        "ml_m": np.zeros((10, 1)),
        "ml_unknown": np.zeros((10, 1)),
    }
    msg = "External predictions provided for unknown learner 'ml_unknown'"
    with pytest.raises(ValueError, match=msg):
        dml_obj.fit_nuisance_models(external_predictions=ext_predictions)


@pytest.mark.ci
def test_scalar_external_predictions_shape():
    """Reject external predictions with incorrect shape."""
    np.random.seed(3141)
    dml_data = make_plr_CCDDHNR2018(n_obs=10)
    dml_obj = PLR(dml_data)
    dml_obj.set_learners(ml_l=Lasso(), ml_m=Lasso())
    dml_obj.draw_sample_splitting(n_folds=2, n_rep=1)

    ext_predictions = {
        "ml_l": np.zeros((10, 2)),
        "ml_m": np.zeros((10, 1)),
    }
    msg = r"External predictions for 'ml_l' must have shape \(10, 1\)"
    with pytest.raises(ValueError, match=msg):
        dml_obj.fit_nuisance_models(external_predictions=ext_predictions)
