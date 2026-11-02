import importlib.util
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.linear_model import LinearRegression

from doubleml import DoubleMLData, DoubleMLPLR

spec = importlib.util.spec_from_file_location("plr_benchmark", Path(__file__).parents[1] / "reproducibility.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


@pytest.fixture
def model():
    rng = np.random.RandomState(12)
    data = DoubleMLData.from_arrays(rng.normal(size=(53, 3)), rng.normal(size=53), rng.normal(size=53))
    return DoubleMLPLR(data, LinearRegression(), LinearRegression(), n_folds=4, draw_sample_splitting=False)


def test_fixed_run_uses_model_dimensions(model, monkeypatch):
    calls = []
    fit = DoubleMLPLR.fit

    def counted_fit(self, *args, **kwargs):
        calls.append((self.n_obs, self.n_folds, self.n_rep))
        return fit(self, *args, **kwargs)

    monkeypatch.setattr(DoubleMLPLR, "fit", counted_fit)
    result = benchmark.fixed_run(model, seed=12, repetitions=3, tolerance=0.01)
    assert calls == [(53, 4, 3)]
    assert np.isfinite(result["mean"])
    assert np.isfinite(result["mcse"])


@pytest.mark.parametrize(
    "exceeded, interval",
    [(False, [0, 0.2775327998628892]), (True, [0.7224672001371107, 1])],
)
def test_benchmark_interval_and_work_count(model, exceeded, interval, monkeypatch):
    def run_pair(*args):
        return {"exceeded": exceeded, "reps": [20, 20], "exhausted": [False, False], "seconds": 1}

    monkeypatch.setattr(benchmark, "make_model", lambda: model)
    monkeypatch.setattr(benchmark, "run_pair", run_pair)
    monkeypatch.setattr(benchmark, "fixed_run", lambda *args: {"mean": 0, "mcse": 0.01, "seconds": 1})
    report = benchmark.benchmark(pairs=10, fixed_runs=1, tolerance=0.01, max_reps=20, n_jobs=1)
    assert report["exceedance_rate"] == float(exceeded)
    assert_allclose(report["exceedance_interval_95"], interval, atol=1e-15)
    assert report["total_nuisance_fits"] == 3200
    for count, result in report["fixed"].items():
        assert result["nuisance_fits_per_run"] == 8 * int(count)
