from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from doubleml import DoubleMLData, DoubleMLPLR
from doubleml.utils import fit_until_reproducible, reproducibility_check

pytestmark = pytest.mark.ci


@pytest.fixture
def model():
    rng = np.random.RandomState(412)
    x = rng.normal(size=(81, 4))
    d = x[:, 0] + rng.normal(size=81)
    y = 0.5 * d + x[:, 1] + rng.normal(size=81)
    data = DoubleMLData.from_arrays(x, y, d)
    return DoubleMLPLR(data, LinearRegression(), LinearRegression(), n_folds=3, draw_sample_splitting=False)


def test_numerical_oracle():
    result = reproducibility_check([1, 2, 3], 2, 0.05)
    assert result["mean"] == 2
    assert_allclose(result["mcse"], np.sqrt(1 / 3))
    assert_allclose(result["threshold"], 2 / (np.sqrt(2) * norm.isf(0.025)))
    assert result["criterion_met"]
    assert result["repetitions"] == 3
    assert reproducibility_check([4, 4], 0.01)["mcse"] == 0
    assert not reproducibility_check([0, 0, 1e6], 0.01)["criterion_met"]


@pytest.mark.parametrize("values", [[], [1], [[1, 2]], [1, np.nan], [1, np.inf], [1j, 2j], ["1", "2"], [True, False]])
def test_invalid_values(values):
    with pytest.raises(ValueError):
        reproducibility_check(values, 0.1)


@pytest.mark.parametrize("name", ["tolerance", "error_rate"])
@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan, True, "0.1", None])
def test_invalid_targets(name, value):
    kwargs = {"tolerance": 0.1, "error_rate": 0.05, name: value}
    with pytest.raises((ValueError, TypeError)):
        reproducibility_check([1, 2], **kwargs)


def test_extreme_targets_and_overflow():
    with pytest.raises(ValueError, match="less than 1"):
        reproducibility_check([1, 2], 0.1, 1)
    assert np.isfinite(reproducibility_check([1, 2], 0.1, 1e-20)["threshold"])
    with pytest.raises(ValueError, match="floating-point precision"):
        reproducibility_check([1, 2], 0.1, np.nextafter(0.0, 1.0))
    with pytest.raises(ValueError, match="overflow"):
        reproducibility_check([-1e308, 1e308], 0.1)


def test_matches_fixed_repetitions_without_mutation(model):
    rng = np.random.RandomState(17)
    splits = [list(KFold(3, shuffle=True, random_state=rng).split(np.empty(81))) for _ in range(5)]
    model.set_sample_splitting(splits[:1]).fit()
    before = deepcopy(model)
    fixed = deepcopy(model)
    fixed.set_sample_splitting(splits).fit()
    report = fit_until_reproducible(model, 1e-14, min_reps=2, max_reps=5, random_state=17)
    assert report["status"] == "budget_exhausted"
    assert_allclose(report["values"], fixed.all_coef[0], rtol=1e-12)
    assert_allclose(report["mean"], fixed.all_coef.mean())
    assert_array_equal(model.all_coef, before.all_coef)
    assert_array_equal(model.se, before.se)
    assert_array_equal(model.pval, before.pval)
    assert_frame_equal(model.summary, before.summary)
    assert_array_equal(model.predictions["ml_l"], before.predictions["ml_l"])
    for actual, expected in zip(model.smpls[0], before.smpls[0]):
        assert_array_equal(actual[0], expected[0])
        assert_array_equal(actual[1], expected[1])


def test_minimum_and_fit_count(model, monkeypatch):
    original_fit = LinearRegression.fit
    calls = []

    def counted(self, x, y, **kwargs):
        calls.append(len(y))
        return original_fit(self, x, y, **kwargs)

    monkeypatch.setattr(LinearRegression, "fit", counted)
    report = fit_until_reproducible(model, 100, min_reps=4, max_reps=9, random_state=10)
    assert report["status"] == "target_reached"
    assert report["repetitions"] == 4
    assert len(calls) == 4 * 3 * 2
    assert report["mcse"] <= report["threshold"]


def test_repeatability_and_tighter_target(model):
    loose = fit_until_reproducible(model, 1, min_reps=3, max_reps=8, random_state=10)
    tight = fit_until_reproducible(model, 1e-12, min_reps=3, max_reps=8, random_state=10)
    again = fit_until_reproducible(model, 1e-12, min_reps=3, max_reps=8, random_state=10)
    assert loose["repetitions"] < tight["repetitions"]
    assert_array_equal(loose["values"], tight["values"][:3])
    assert_array_equal(tight["values"], again["values"])


@pytest.mark.parametrize("kwargs", [{"min_reps": 1}, {"min_reps": True}, {"max_reps": 2}, {"max_reps": 2.5}])
def test_invalid_budget(model, kwargs):
    with pytest.raises((TypeError, ValueError)):
        fit_until_reproducible(model, 0.1, **kwargs)


def test_unsupported_models(model):
    with pytest.raises(TypeError, match="DoubleMLPLR"):
        fit_until_reproducible(object(), 0.1)
    model._n_rep = 2
    with pytest.raises(ValueError, match="n_rep=1"):
        fit_until_reproducible(model, 0.1)


def test_reject_fold_params_and_unseeded_learner(model):
    model.set_ml_nuisance_params("ml_l", "d", {"fit_intercept": False})
    with pytest.raises(ValueError, match="Fold-specific"):
        fit_until_reproducible(model, 0.1)
    model = DoubleMLPLR(model._dml_data, RandomForestRegressor(), LinearRegression())
    with pytest.raises(ValueError, match="random_state"):
        fit_until_reproducible(model, 0.1)


@pytest.mark.parametrize("attribute,value", [("_is_cluster_data", True), ("_n_folds", 1), ("_score", lambda *args: None)])
def test_unsupported_configuration(model, attribute, value):
    setattr(model, attribute, value)
    with pytest.raises(ValueError):
        fit_until_reproducible(model, 0.1)


def test_multiple_treatments_rejected(model):
    data = model._dml_data
    multiple = DoubleMLData.from_arrays(data.x, data.y, np.column_stack([data.d, data.x[:, 2]]))
    model = DoubleMLPLR(multiple, LinearRegression(), LinearRegression())
    with pytest.raises(ValueError, match="one treatment"):
        fit_until_reproducible(model, 0.1)


def test_nonfinite_fit_fails_immediately(model, monkeypatch):
    calls = []

    def invalid_fit(self, **kwargs):
        calls.append(1)
        self._all_coef[:] = np.nan
        return self

    monkeypatch.setattr(DoubleMLPLR, "fit", invalid_fit)
    with pytest.raises(ValueError, match="non-finite coefficient"):
        fit_until_reproducible(model, 0.1)
    assert len(calls) == 1


def test_iv_type_score(model):
    template = DoubleMLPLR(
        model._dml_data,
        LinearRegression(),
        LinearRegression(),
        ml_g=LinearRegression(),
        score="IV-type",
        n_folds=3,
        draw_sample_splitting=False,
    )
    report = fit_until_reproducible(template, 1, min_reps=3, max_reps=3, random_state=7)
    assert report["status"] == "target_reached"
    assert np.isfinite(report["values"]).all()


def test_report_contract(model):
    report = fit_until_reproducible(model, 1, min_reps=3, max_reps=3, random_state=7)
    assert isinstance(report, dict)
    assert set(report) == {
        "mean",
        "mcse",
        "threshold",
        "criterion_met",
        "repetitions",
        "tolerance",
        "error_rate",
        "status",
        "values",
    }
    for name in ("mean", "mcse", "threshold", "tolerance", "error_rate"):
        assert isinstance(report[name], float)
    assert isinstance(report["criterion_met"], bool)
    assert isinstance(report["repetitions"], int)
    assert isinstance(report["values"], np.ndarray)
    assert report["values"].shape == (report["repetitions"],)
    assert report["values"].dtype.kind == "f"
