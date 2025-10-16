import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from doubleml import DoubleMLIRM, DoubleMLPLR
from doubleml.irm.datasets import make_irm_data
from doubleml.plm.datasets import make_plr_CCDDHNR2018

# Test setup
n_folds = 3
n_rep = 2

np.random.seed(3141)
dml_data_irm = make_irm_data(n_obs=1000)

reg_learner = RandomForestRegressor(max_depth=2, n_estimators=100)
class_learner = RandomForestClassifier(max_depth=2, n_estimators=100)


@pytest.fixture
def fresh_irm_model():
    """Create a fresh IRM model for each test."""
    return DoubleMLIRM(dml_data_irm, reg_learner, class_learner, n_folds=n_folds, n_rep=n_rep)


@pytest.mark.ci
def test_set_single_params(fresh_irm_model):
    """Test combining behavior where new parameters are merged with existing ones."""
    dml_irm = fresh_irm_model

    # Set initial parameters
    initial_params = {"n_estimators": 50, "max_depth": 3}
    dml_irm.set_ml_nuisance_params("ml_g0", "d", initial_params)

    # Set additional parameters (should combine)
    additional_params = {"min_samples_split": 5, "n_estimators": 25}  # n_estimators should be updated
    dml_irm.set_ml_nuisance_params("ml_g0", "d", additional_params)

    # With combining behavior, we should have all keys
    expected_combined = {"n_estimators": 25, "max_depth": 3, "min_samples_split": 5}
    assert dml_irm.params["ml_g0"]["d"][0][0] == expected_combined
    assert dml_irm.params["ml_g0"]["d"][1][1] == expected_combined


@pytest.mark.ci
def test_none_params_handling(fresh_irm_model):
    """Test handling of None parameters."""
    dml_irm = fresh_irm_model

    # Set initial parameters
    initial_params = {"n_estimators": 50}
    dml_irm.set_ml_nuisance_params("ml_g0", "d", initial_params)

    # Setting None should not change existing parameters
    dml_irm.set_ml_nuisance_params("ml_g0", "d", None)
    assert dml_irm.params["ml_g0"]["d"][0][0] == initial_params

    # Test setting None on empty parameters
    dml_irm.set_ml_nuisance_params("ml_g1", "d", None)
    assert dml_irm.params["ml_g1"]["d"] == [None] * n_rep


@pytest.mark.ci
def test_set_nested_list_params(fresh_irm_model):
    """Test combining behavior with nested list parameters."""
    dml_irm = fresh_irm_model

    # Create initial nested parameters
    initial_nested = [
        [
            {"n_estimators": 50, "max_depth": 2},
            {"n_estimators": 60, "max_depth": 3},
            {"n_estimators": 60, "max_depth": 3},
        ],  # rep 0
        [
            {"n_estimators": 70, "max_depth": 4},
            {"n_estimators": 80, "max_depth": 5},
            {"n_estimators": 60, "max_depth": 3},
        ],  # rep 1
    ]
    dml_irm.set_ml_nuisance_params("ml_g0", "d", initial_nested)

    # Add additional parameters
    additional_nested = [
        [
            {"min_samples_split": 2, "n_estimators": 25},
            {"min_samples_split": 3, "n_estimators": 35},
            {"min_samples_split": 3, "n_estimators": 35},
        ],  # rep 0
        [
            {"min_samples_split": 4, "n_estimators": 45},
            {"min_samples_split": 5, "n_estimators": 55},
            {"min_samples_split": 3, "n_estimators": 35},
        ],  # rep 1
    ]
    dml_irm.set_ml_nuisance_params("ml_g0", "d", additional_nested)

    # Verify combining: existing keys preserved, overlapping keys updated, new keys added
    expected_combined = [
        [
            {"n_estimators": 25, "max_depth": 2, "min_samples_split": 2},
            {"n_estimators": 35, "max_depth": 3, "min_samples_split": 3},
            {"n_estimators": 35, "max_depth": 3, "min_samples_split": 3},
        ],
        [
            {"n_estimators": 45, "max_depth": 4, "min_samples_split": 4},
            {"n_estimators": 55, "max_depth": 5, "min_samples_split": 5},
            {"n_estimators": 35, "max_depth": 3, "min_samples_split": 3},
        ],
    ]

    assert dml_irm.params["ml_g0"]["d"] == expected_combined


@pytest.mark.ci
def test_multiple_learners_independence(fresh_irm_model):
    """Test that parameters for different learners are independent."""
    dml_irm = fresh_irm_model

    # Set parameters for different learners
    params_g0 = {"n_estimators": 50}
    params_g1 = {"n_estimators": 75}
    params_m = {"n_estimators": 100}

    dml_irm.set_ml_nuisance_params("ml_g0", "d", params_g0)
    dml_irm.set_ml_nuisance_params("ml_g1", "d", params_g1)
    dml_irm.set_ml_nuisance_params("ml_m", "d", params_m)

    # Verify independence
    assert dml_irm.params["ml_g0"]["d"][0][0] == params_g0
    assert dml_irm.params["ml_g1"]["d"][0][0] == params_g1
    assert dml_irm.params["ml_m"]["d"][0][0] == params_m

    # Modify one learner, others should remain unchanged
    new_params_g0 = {"max_depth": 3, "n_estimators": 25}
    dml_irm.set_ml_nuisance_params("ml_g0", "d", new_params_g0)

    # With combining behavior
    expected_g0 = {"n_estimators": 25, "max_depth": 3}
    assert dml_irm.params["ml_g0"]["d"][0][0] == expected_g0
    assert dml_irm.params["ml_g1"]["d"][0][0] == params_g1  # unchanged
    assert dml_irm.params["ml_m"]["d"][0][0] == params_m  # unchanged


@pytest.mark.ci
def test_multiple_treatment_variables_independence():
    """Test that parameters for different treatment variables are independent."""
    # Create PLR data with multiple treatment variables
    np.random.seed(3141)
    multi_treat_data = make_plr_CCDDHNR2018(n_obs=100)

    # Add a second treatment variable for testing
    multi_treat_data.data["d2"] = np.random.normal(0, 1, 100)
    multi_treat_data._d_cols = ["d", "d2"]

    dml_plr = DoubleMLPLR(multi_treat_data, reg_learner, reg_learner, n_folds=n_folds, n_rep=n_rep)

    # Set parameters for different treatment variables
    params_d = {"n_estimators": 50}
    params_d2 = {"n_estimators": 75}

    dml_plr.set_ml_nuisance_params("ml_l", "d", params_d)
    dml_plr.set_ml_nuisance_params("ml_l", "d2", params_d2)

    # Verify independence
    assert dml_plr.params["ml_l"]["d"][0][0] == params_d
    assert dml_plr.params["ml_l"]["d2"][0][0] == params_d2

    # Modify one treatment variable, other should remain unchanged
    new_params_d = {"max_depth": 3, "n_estimators": 25}
    dml_plr.set_ml_nuisance_params("ml_l", "d", new_params_d)

    # With combining behavior
    expected_d = {"n_estimators": 25, "max_depth": 3}
    assert dml_plr.params["ml_l"]["d"][0][0] == expected_d
    assert dml_plr.params["ml_l"]["d2"][0][0] == params_d2  # unchanged


@pytest.mark.ci
def test_error_cases(fresh_irm_model):
    """Test error handling for invalid inputs."""
    dml_irm = fresh_irm_model

    # Invalid learner
    with pytest.raises(ValueError, match="Invalid nuisance learner"):
        dml_irm.set_ml_nuisance_params("invalid_learner", "d", {"n_estimators": 50})

    # Invalid treatment variable
    with pytest.raises(ValueError, match="Invalid treatment variable"):
        dml_irm.set_ml_nuisance_params("ml_g0", "invalid_treat", {"n_estimators": 50})

    # Invalid nested list length (wrong n_rep)
    invalid_nested = [[{"n_estimators": 50}, {"n_estimators": 60}]]  # Only 1 rep, should be 2
    with pytest.raises(AssertionError):
        dml_irm.set_ml_nuisance_params("ml_g0", "d", invalid_nested)

    # Invalid nested list length (wrong n_folds)
    invalid_nested = [[{"n_estimators": 50}], [{"n_estimators": 60}]]  # Only 1 fold, should be 2  # Only 1 fold, should be 2
    with pytest.raises(AssertionError):
        dml_irm.set_ml_nuisance_params("ml_g0", "d", invalid_nested)
