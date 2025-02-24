import numpy as np
import pytest
from sklearn.linear_model import Lasso, LogisticRegression

from doubleml.data import DoubleMLData
from doubleml.did import DoubleMLDID, DoubleMLDIDCS
from doubleml.did.datasets import make_did_SZ2020
from doubleml.utils._check_return_types import (
    check_basic_predictions_and_targets,
    check_basic_property_types_and_shapes,
    check_basic_return_types,
    check_sensitivity_return_types,
)

# Test constants
N_OBS = 200
N_TREAT = 1
N_REP = 1
N_FOLDS = 3
N_REP_BOOT = 314

dml_args = {
    "n_rep": N_REP,
    "n_folds": N_FOLDS,
}

# create all datasets
np.random.seed(3141)
datasets = {}

datasets["did"] = make_did_SZ2020(n_obs=N_OBS)
datasets["did_cs"] = make_did_SZ2020(n_obs=N_OBS, cross_sectional_data=True)

# Binary outcome
(x, y, d, t) = make_did_SZ2020(n_obs=N_OBS, cross_sectional_data=True, return_type="array")
binary_outcome = np.random.binomial(n=1, p=0.5, size=N_OBS)
datasets["did_binary_outcome"] = DoubleMLData.from_arrays(x, binary_outcome, d)
datasets["did_cs_binary_outcome"] = DoubleMLData.from_arrays(x, binary_outcome, d, t=t)

dml_objs = [
    (DoubleMLDID(datasets["did"], Lasso(), LogisticRegression(), **dml_args), DoubleMLDID),
    (DoubleMLDID(datasets["did_binary_outcome"], LogisticRegression(), LogisticRegression(), **dml_args), DoubleMLDID),
    (DoubleMLDIDCS(datasets["did_cs"], Lasso(), LogisticRegression(), **dml_args), DoubleMLDIDCS),
    (DoubleMLDIDCS(datasets["did_cs_binary_outcome"], LogisticRegression(), LogisticRegression(), **dml_args), DoubleMLDIDCS),
]


@pytest.mark.ci
@pytest.mark.parametrize("dml_obj, cls", dml_objs)
def test_return_types(dml_obj, cls):
    check_basic_return_types(dml_obj, cls)

    # further return type tests
    assert isinstance(dml_obj.get_params("ml_m"), dict)


@pytest.fixture(params=dml_objs)
def fitted_dml_obj(request):
    dml_obj, _ = request.param
    dml_obj.fit()
    dml_obj.bootstrap(n_rep_boot=N_REP_BOOT)
    return dml_obj


@pytest.mark.ci
def test_property_types_and_shapes(fitted_dml_obj):
    check_basic_property_types_and_shapes(fitted_dml_obj, N_OBS, N_TREAT, N_REP, N_FOLDS, N_REP_BOOT)
    check_basic_predictions_and_targets(fitted_dml_obj, N_OBS, N_TREAT, N_REP)


@pytest.mark.ci
def test_sensitivity_return_types(fitted_dml_obj):
    if fitted_dml_obj._sensitivity_implemented:
        benchmarking_set = [fitted_dml_obj._dml_data.x_cols[0]]
        check_sensitivity_return_types(fitted_dml_obj, N_OBS, N_REP, N_TREAT, benchmarking_set=benchmarking_set)
