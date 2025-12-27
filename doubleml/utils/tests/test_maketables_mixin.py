"""
Tests for MakeTables Mixin.

This module tests the MakeTables plug-in support for DoubleML models,
verifying that the mixin correctly provides coefficient tables, statistics,
and dependent variable names for use with the MakeTables package.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Lasso

import doubleml as dml


@pytest.fixture(scope="module")
def generate_plr_data():
    """Generate simple data for PLR model testing."""
    np.random.seed(42)
    n = 500
    p = 5
    theta = 0.5

    # Generate simple data
    X = np.random.normal(size=(n, p))
    d = 0.5 * X[:, 0] + np.random.normal(size=n)
    y = theta * d + X[:, 1] + np.random.normal(size=n)

    df = pd.DataFrame(
        np.column_stack((X, y, d)),
        columns=[f"X{i+1}" for i in range(p)] + ["Y", "D"]
    )

    return dml.DoubleMLData(df, "Y", "D")


@pytest.fixture(scope="module")
def fitted_plr_model(generate_plr_data):
    """Create and fit a simple PLR model for testing."""
    ml_l = LinearRegression()
    ml_m = LinearRegression()

    dml_plr = dml.DoubleMLPLR(
        generate_plr_data,
        ml_l,
        ml_m,
        n_folds=2,
        score="partialling out"
    )
    dml_plr.fit()

    return dml_plr


@pytest.fixture(scope="module")
def unfitted_plr_model(generate_plr_data):
    """Create an unfitted PLR model for testing edge cases."""
    ml_l = LinearRegression()
    ml_m = LinearRegression()

    dml_plr = dml.DoubleMLPLR(
        generate_plr_data,
        ml_l,
        ml_m,
        n_folds=2,
        score="partialling out"
    )

    return dml_plr


@pytest.fixture(scope="module")
def generate_irm_data():
    """Generate simple data for IRM model testing."""
    np.random.seed(43)
    n = 500
    p = 5

    # Generate simple data with binary treatment
    X = np.random.normal(size=(n, p))
    propensity = 1 / (1 + np.exp(-X[:, 0]))
    d = (np.random.uniform(size=n) < propensity).astype(float)
    y = 0.5 * d + X[:, 1] + np.random.normal(size=n)

    df = pd.DataFrame(
        np.column_stack((X, y, d)),
        columns=[f"X{i+1}" for i in range(p)] + ["Y", "D"]
    )

    return dml.DoubleMLData(df, "Y", "D")


@pytest.fixture(scope="module")
def fitted_irm_model(generate_irm_data):
    """Create and fit a simple IRM model for testing."""
    from sklearn.linear_model import LogisticRegression

    ml_g = LinearRegression()
    ml_m = LogisticRegression()

    dml_irm = dml.DoubleMLIRM(
        generate_irm_data,
        ml_g,
        ml_m,
        n_folds=2,
        score="ATE"
    )
    dml_irm.fit()

    return dml_irm


# ==================================================================================
# Test Coefficient Table Structure
# ==================================================================================


@pytest.mark.ci
def test_coef_table_has_required_columns(fitted_plr_model):
    """Test that coefficient table has all required columns."""
    coef_table = fitted_plr_model.__maketables_coef_table__

    # Check DataFrame type
    assert isinstance(coef_table, pd.DataFrame)

    # Check required columns exist
    assert "b" in coef_table.columns, "Missing required column 'b'"
    assert "se" in coef_table.columns, "Missing required column 'se'"
    assert "p" in coef_table.columns, "Missing required column 'p'"


@pytest.mark.ci
def test_coef_table_has_optional_columns(fitted_plr_model):
    """Test that coefficient table has optional columns."""
    coef_table = fitted_plr_model.__maketables_coef_table__

    # Check optional columns exist
    assert "t" in coef_table.columns, "Missing optional column 't'"
    assert "ci95l" in coef_table.columns, "Missing optional column 'ci95l'"
    assert "ci95u" in coef_table.columns, "Missing optional column 'ci95u'"


@pytest.mark.ci
def test_coef_table_index_matches_summary(fitted_plr_model):
    """Test that coefficient table index matches summary index."""
    coef_table = fitted_plr_model.__maketables_coef_table__
    summary = fitted_plr_model.summary

    # Index should match treatment variable names
    assert list(coef_table.index) == list(summary.index)


# ==================================================================================
# Test Coefficient Table Values
# ==================================================================================


@pytest.mark.ci
def test_coef_table_values_match_model(fitted_plr_model):
    """Test that coefficient table values match the model's estimates."""
    coef_table = fitted_plr_model.__maketables_coef_table__

    # Check coefficient estimates
    np.testing.assert_array_almost_equal(
        coef_table["b"].values,
        fitted_plr_model.coef,
        decimal=10,
        err_msg="Coefficient estimates don't match"
    )

    # Check standard errors
    np.testing.assert_array_almost_equal(
        coef_table["se"].values,
        fitted_plr_model.se,
        decimal=10,
        err_msg="Standard errors don't match"
    )

    # Check t-statistics
    np.testing.assert_array_almost_equal(
        coef_table["t"].values,
        fitted_plr_model.t_stat,
        decimal=10,
        err_msg="T-statistics don't match"
    )

    # Check p-values
    np.testing.assert_array_almost_equal(
        coef_table["p"].values,
        fitted_plr_model.pval,
        decimal=10,
        err_msg="P-values don't match"
    )


@pytest.mark.ci
def test_coef_table_confidence_intervals(fitted_plr_model):
    """Test that confidence intervals match confint() method."""
    coef_table = fitted_plr_model.__maketables_coef_table__
    ci = fitted_plr_model.confint(level=0.95)

    # Check lower CI bound
    np.testing.assert_array_almost_equal(
        coef_table["ci95l"].values,
        ci.iloc[:, 0].values,
        decimal=10,
        err_msg="Lower CI bounds don't match"
    )

    # Check upper CI bound
    np.testing.assert_array_almost_equal(
        coef_table["ci95u"].values,
        ci.iloc[:, 1].values,
        decimal=10,
        err_msg="Upper CI bounds don't match"
    )


# ==================================================================================
# Test Statistics Method
# ==================================================================================


@pytest.mark.ci
def test_stat_method_returns_n_obs(fitted_plr_model):
    """Test that __maketables_stat__ returns number of observations for key 'N'."""
    n_obs = fitted_plr_model.__maketables_stat__("N")

    assert n_obs is not None, "Should return number of observations"
    assert n_obs == fitted_plr_model.n_obs, "N should match model's n_obs"
    assert isinstance(n_obs, (int, np.integer)), "N should be an integer"


@pytest.mark.ci
def test_stat_method_unknown_keys_return_none(fitted_plr_model):
    """Test that __maketables_stat__ returns None for unknown keys."""
    # Unknown key should return None
    assert fitted_plr_model.__maketables_stat__("unknown_key") is None

    # Empty string should return None
    assert fitted_plr_model.__maketables_stat__("") is None


@pytest.mark.ci
def test_stat_method_traditional_stats_return_none(fitted_plr_model):
    """Test that traditional stats (r2, aic, bic) return None for causal models."""
    # R-squared not applicable for causal inference
    assert fitted_plr_model.__maketables_stat__("r2") is None
    assert fitted_plr_model.__maketables_stat__("adj_r2") is None

    # Information criteria not applicable
    assert fitted_plr_model.__maketables_stat__("aic") is None
    assert fitted_plr_model.__maketables_stat__("bic") is None

    # Log-likelihood not applicable
    assert fitted_plr_model.__maketables_stat__("ll") is None


# ==================================================================================
# Test Dependent Variable
# ==================================================================================


@pytest.mark.ci
def test_depvar_returns_string(fitted_plr_model):
    """Test that __maketables_depvar__ returns a string."""
    depvar = fitted_plr_model.__maketables_depvar__

    assert isinstance(depvar, str), "Dependent variable name should be a string"


@pytest.mark.ci
def test_depvar_matches_data(fitted_plr_model):
    """Test that __maketables_depvar__ matches the actual dependent variable."""
    depvar = fitted_plr_model.__maketables_depvar__

    assert depvar == "Y", "Dependent variable should be 'Y'"
    assert depvar == fitted_plr_model._dml_data.y_col, "Should match data's y_col"


# ==================================================================================
# Test Default Statistics Keys
# ==================================================================================


@pytest.mark.ci
def test_default_stat_keys_returns_list(fitted_plr_model):
    """Test that __maketables_default_stat_keys__ returns a list."""
    default_keys = fitted_plr_model.__maketables_default_stat_keys__

    assert isinstance(default_keys, list), "Default stat keys should be a list"


@pytest.mark.ci
def test_default_stat_keys_contains_n(fitted_plr_model):
    """Test that default statistics include 'N'."""
    default_keys = fitted_plr_model.__maketables_default_stat_keys__

    assert "N" in default_keys, "Default statistics should include 'N'"


# ==================================================================================
# Test Multiple Model Types
# ==================================================================================


@pytest.mark.ci
def test_maketables_works_with_irm_model(fitted_irm_model):
    """Test that maketables mixin works with IRM models."""
    # Should have coefficient table
    coef_table = fitted_irm_model.__maketables_coef_table__
    assert isinstance(coef_table, pd.DataFrame)
    assert "b" in coef_table.columns
    assert "se" in coef_table.columns
    assert "p" in coef_table.columns

    # Should return N statistic
    assert fitted_irm_model.__maketables_stat__("N") == fitted_irm_model.n_obs

    # Should return depvar
    assert fitted_irm_model.__maketables_depvar__ == "Y"


@pytest.mark.ci
def test_maketables_works_with_pliv_model(generate_plr_data):
    """Test that maketables mixin works with PLIV models."""
    from doubleml.plm.datasets import make_pliv_CHS2015

    # Generate IV data
    np.random.seed(44)
    data = make_pliv_CHS2015(n_obs=500, dim_x=5, alpha=0.5, dim_z=1, return_type=pd.DataFrame)

    x_cols = [col for col in data.columns if col.startswith("X")]
    dml_data = dml.DoubleMLData(data, "y", "d", x_cols, z_cols="Z1")

    ml_l = LinearRegression()
    ml_m = LinearRegression()
    ml_r = LinearRegression()

    dml_pliv = dml.DoubleMLPLIV(dml_data, ml_l, ml_m, ml_r, n_folds=2)
    dml_pliv.fit()

    # Should have coefficient table
    coef_table = dml_pliv.__maketables_coef_table__
    assert isinstance(coef_table, pd.DataFrame)
    assert "b" in coef_table.columns

    # Should return N statistic
    assert dml_pliv.__maketables_stat__("N") == dml_pliv.n_obs

    # Should return depvar
    assert dml_pliv.__maketables_depvar__ == "y"


# ==================================================================================
# Test Edge Cases
# ==================================================================================


@pytest.mark.ci
def test_unfitted_model_returns_empty_dataframe(unfitted_plr_model):
    """Test that unfitted model returns empty DataFrame with correct columns."""
    coef_table = unfitted_plr_model.__maketables_coef_table__

    assert isinstance(coef_table, pd.DataFrame), "Should return DataFrame"
    assert len(coef_table) == 0, "Should be empty for unfitted model"

    # Should still have correct columns
    expected_columns = ["b", "se", "t", "p", "ci95l", "ci95u"]
    assert list(coef_table.columns) == expected_columns


@pytest.mark.ci
def test_unfitted_model_stat_returns_n(unfitted_plr_model):
    """Test that unfitted model can still return N statistic."""
    n_obs = unfitted_plr_model.__maketables_stat__("N")

    # Should still have n_obs even if not fitted
    assert n_obs is not None
    assert n_obs == unfitted_plr_model.n_obs


@pytest.mark.ci
def test_unfitted_model_depvar_works(unfitted_plr_model):
    """Test that unfitted model can return dependent variable name."""
    depvar = unfitted_plr_model.__maketables_depvar__

    assert depvar == "Y"


@pytest.mark.ci
def test_multi_treatment_model():
    """Test that maketables works with multiple treatment variables."""
    np.random.seed(45)
    n = 500
    p = 5

    # Generate data with 2 treatments
    X = np.random.normal(size=(n, p))
    D1 = 0.5 * X[:, 0] + np.random.normal(size=n)
    D2 = 0.3 * X[:, 1] + np.random.normal(size=n)
    Y = 0.5 * D1 + 0.7 * D2 + X[:, 2] + np.random.normal(size=n)

    df = pd.DataFrame(
        np.column_stack((X, Y, D1, D2)),
        columns=[f"X{i+1}" for i in range(p)] + ["Y", "D1", "D2"]
    )

    dml_data = dml.DoubleMLData(df, "Y", ["D1", "D2"])

    ml_l = LinearRegression()
    ml_m = LinearRegression()

    dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=2, score="partialling out")
    dml_plr.fit()

    # Coefficient table should have 2 rows (one per treatment)
    coef_table = dml_plr.__maketables_coef_table__
    assert len(coef_table) == 2, "Should have 2 rows for 2 treatments"
    assert list(coef_table.index) == ["D1", "D2"], "Index should be treatment names"


# ==================================================================================
# Test Integration (Optional - requires maketables)
# ==================================================================================


@pytest.mark.ci
def test_integration_with_maketables_if_available(fitted_plr_model):
    """Test integration with maketables package if it's installed."""
    pytest.importorskip("maketables", reason="maketables not installed")

    from maketables import ETable

    # Should be able to create a table
    try:
        table = ETable([fitted_plr_model])
        assert table is not None

        # Should be able to render
        text_output = table.render("txt")
        assert isinstance(text_output, str)
        assert len(text_output) > 0

    except Exception as e:
        pytest.fail(f"MakeTables integration failed: {str(e)}")


# ==================================================================================
# Test Comparison with Summary
# ==================================================================================


@pytest.mark.ci
def test_coef_table_consistent_with_summary(fitted_plr_model):
    """Test that coefficient table is consistent with summary property."""
    coef_table = fitted_plr_model.__maketables_coef_table__
    summary = fitted_plr_model.summary

    # Same index
    assert list(coef_table.index) == list(summary.index)

    # Coefficients match
    np.testing.assert_array_almost_equal(
        coef_table["b"].values,
        summary["coef"].values,
        decimal=10
    )

    # Standard errors match
    np.testing.assert_array_almost_equal(
        coef_table["se"].values,
        summary["std err"].values,
        decimal=10
    )

    # T-statistics match
    np.testing.assert_array_almost_equal(
        coef_table["t"].values,
        summary["t"].values,
        decimal=10
    )

    # P-values match
    np.testing.assert_array_almost_equal(
        coef_table["p"].values,
        summary["P>|t|"].values,
        decimal=10
    )
