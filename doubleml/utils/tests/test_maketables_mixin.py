"""Test MakeTables integration against fitted DoubleML estimates and rendered tables."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from maketables import ETable
from maketables.extractors import get_extractor
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml


@pytest.fixture(scope="module", params=["plr", "multi", "irm", "pliv", "cluster"])
def model_kind(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module")
def fitted_model(model_kind, n_rep):
    np.random.seed(3141)
    n_obs = 200
    x = np.random.normal(size=(n_obs, 3))
    z = np.random.normal(size=n_obs)
    d = x[:, 0] + z + np.random.normal(size=n_obs)
    if model_kind == "irm":
        d = (d > 0).astype(float)
    second = x[:, 1] + np.random.normal(size=n_obs)
    y = 0.5 * d + 0.3 * second + x[:, 2] + np.random.normal(size=n_obs)
    frame = pd.DataFrame(x, columns=["x1", "x2", "x3"])
    frame["outcome"] = y
    frame["treatment_z"] = d
    frame["treatment_a"] = second
    frame["instrument"] = z
    d_cols = ["treatment_z", "treatment_a"] if model_kind == "multi" else ["treatment_z"]
    kwargs = {"y_col": "outcome", "d_cols": d_cols, "x_cols": ["x1", "x2", "x3"]}
    if model_kind == "cluster":
        frame["cluster"] = np.repeat(np.arange(40), 5)
        data = dml.DoubleMLData(frame, cluster_cols="cluster", **kwargs)
    else:
        data = dml.DoubleMLData(frame, z_cols="instrument" if model_kind == "pliv" else None, **kwargs)
    if model_kind == "irm":
        model = dml.DoubleMLIRM(data, LinearRegression(), LogisticRegression(), n_folds=2, n_rep=n_rep)
    elif model_kind == "pliv":
        model = dml.DoubleMLPLIV(data, LinearRegression(), LinearRegression(), LinearRegression(), n_folds=2, n_rep=n_rep)
    else:
        model = dml.DoubleMLPLR(data, LinearRegression(), LinearRegression(), n_folds=2, n_rep=n_rep)
    return model.fit()


@pytest.mark.ci
def test_maketables_estimates(fitted_model):
    """The detected extractor preserves estimates, inference, and treatment order."""
    model = fitted_model
    before = model.summary.copy(deep=True)
    table = get_extractor(model).coef_table(model)
    expected = before.rename(columns={"coef": "b", "std err": "se", "P>|t|": "p", "2.5 %": "ci95l", "97.5 %": "ci95u"})
    expected.index.name = "Coefficient"
    pd.testing.assert_frame_equal(table[expected.columns], expected)
    pd.testing.assert_frame_equal(model.summary, before)
    table.iloc[0, 0] = 999
    pd.testing.assert_frame_equal(model.summary, before)


@pytest.mark.ci
def test_maketables_metadata(fitted_model):
    """MakeTables receives the outcome name, sample size, and supported defaults."""
    extractor = get_extractor(fitted_model)
    assert extractor.depvar(fitted_model) == "outcome"
    assert extractor.stat(fitted_model, "N") == 200
    assert extractor.default_stat_keys(fitted_model) == ["N"]
    assert extractor.stat(fitted_model, "r2") is None
    assert extractor.stat(fitted_model, "unknown") is None


@pytest.mark.ci
@pytest.mark.parametrize("output_format", ["tex", "html"])
def test_maketables_render(fitted_model, output_format):
    """Actual exported tables contain estimates and pointwise confidence intervals."""
    model = fitted_model
    table = ETable(model, coef_fmt="b:.6f (se:.6f) [ci95l:.6f, ci95u:.6f]", labels={"treatment_z": "Treatment"})
    output = table.make(output_format)
    assert "Treatment" in output
    assert "outcome" in output
    assert "200" in output
    for value in [model.coef[0], model.se[0], *model.confint().iloc[0]]:
        assert f"{value:.6f}" in output


@pytest.mark.ci
def test_maketables_multiple_models(fitted_model):
    """The same estimate is rendered in each requested model column."""
    output = ETable([fitted_model, fitted_model], coef_fmt="b:.6f").make("tex")
    assert output.count(f"{fitted_model.coef[0]:.6f}") == 2


@pytest.mark.ci
def test_maketables_before_fit():
    """An unfitted model produces no estimates, matching DoubleML.summary."""
    np.random.seed(3141)
    data = dml.DoubleMLData.from_arrays(np.random.normal(size=(20, 2)), np.arange(20.0), np.arange(20.0))
    model = dml.DoubleMLPLR(data, LinearRegression(), LinearRegression(), n_folds=2)
    table = model.__maketables_coef_table__
    assert table.empty
    assert list(table.columns) == ["b", "se", "t", "p", "ci95l", "ci95u"]
    output = ETable(model).make("tex")
    assert "20" in output
    assert "nan" not in output.lower()
    assert model.framework is None


@pytest.mark.ci
def test_maketables_reset_and_refit(fitted_model):
    """Resetting sample splits must hide old intervals until the model is refitted."""
    model = deepcopy(fitted_model)
    before = model.__maketables_coef_table__
    model.set_sample_splitting(model.smpls, model.smpls_cluster)
    assert model.summary.empty
    assert model.__maketables_coef_table__.empty
    output = ETable(model, coef_fmt="b:.6f [ci95l:.6f, ci95u:.6f]").make("tex")
    for value in before["ci95l"]:
        assert f"{value:.6f}" not in output
    model.fit()
    pd.testing.assert_frame_equal(model.__maketables_coef_table__, before)
