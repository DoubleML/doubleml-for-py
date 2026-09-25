import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso, LogisticRegression

from doubleml import DoubleMLIRM, DoubleMLPolicyTree
from doubleml.irm.datasets import make_irm_data
from doubleml.utils import PSProcessorConfig
from doubleml.utils._check_return_types import (
    check_basic_predictions_and_targets,
    check_basic_property_types_and_shapes,
    check_basic_return_types,
    check_sensitivity_return_types,
)

N_OBS = 200
N_TREAT = 1
N_REP = 2
N_FOLDS = 3
N_REP_BOOT = 314

np.random.seed(3141)
dml_data_irm = make_irm_data(n_obs=N_OBS)

dml_objs = [(DoubleMLIRM(dml_data_irm, Lasso(), LogisticRegression()), DoubleMLIRM)]


@pytest.mark.ci
@pytest.mark.parametrize("dml_obj, cls", dml_objs)
def test_return_types(dml_obj, cls):
    """Check public return types for the legacy DoubleMLIRM API."""
    check_basic_return_types(dml_obj, cls)
    assert isinstance(dml_obj.get_params("ml_m"), dict)


fitted_dml_objs = [
    DoubleMLIRM(
        dml_data_irm,
        Lasso(),
        LogisticRegression(),
        n_rep=N_REP,
        n_folds=N_FOLDS,
        ps_processor_config=PSProcessorConfig(clipping_threshold=0.1),
    )
]


@pytest.fixture(params=fitted_dml_objs)
def fitted_dml_obj(request):
    """Fit and bootstrap a legacy DoubleMLIRM object for shared contract checks."""
    dml_obj = request.param
    dml_obj.fit()
    dml_obj.bootstrap(n_rep_boot=N_REP_BOOT)
    return dml_obj


@pytest.mark.ci
def test_property_types_and_shapes(fitted_dml_obj):
    """Check fitted property types, shapes, predictions, targets, and losses."""
    check_basic_property_types_and_shapes(fitted_dml_obj, N_OBS, N_TREAT, N_REP, N_FOLDS, N_REP_BOOT)
    check_basic_predictions_and_targets(fitted_dml_obj, N_OBS, N_TREAT, N_REP)


@pytest.mark.ci
def test_sensitivity_return_types(fitted_dml_obj):
    """Check sensitivity-analysis return types for the fitted IRM object."""
    check_sensitivity_return_types(fitted_dml_obj, N_OBS, N_REP, N_TREAT, benchmarking_set=["X1"])


@pytest.mark.ci
def test_policytree():
    """Check policy-tree construction and prediction return types."""
    dml_irm = DoubleMLIRM(dml_data_irm, Lasso(), LogisticRegression())
    dml_irm.fit()
    features = dml_data_irm.data[["X1", "X2"]]
    policy_tree = dml_irm.policy_tree(features, depth=2)

    assert isinstance(policy_tree, DoubleMLPolicyTree)
    predict_features = pd.DataFrame(np.random.normal(size=(5, 2)), columns=features.keys())
    assert isinstance(policy_tree.predict(predict_features), pd.DataFrame)
