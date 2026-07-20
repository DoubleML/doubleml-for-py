"""Test cate() and gate() for the PLR scalar model."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso

import doubleml as dml
from doubleml.plm.datasets import make_plr_CCDDHNR2018
from doubleml.plm.plr_scalar import PLR
from doubleml.utils.blp import DoubleMLBLP

N_OBS = 200
N_FOLDS = 3
BASIS_DIM = 4


def _build_plr(n_rep: int, score: str = "partialling out", random_state: int = 42) -> tuple[PLR, pd.DataFrame]:
    """Build and fit a PLR scalar model with a random basis."""
    np.random.seed(random_state)
    data = make_plr_CCDDHNR2018(n_obs=N_OBS, dim_x=5, alpha=0.5, return_type="DoubleMLData")

    ml_l = Lasso(alpha=0.1)
    ml_m = Lasso(alpha=0.1)
    ml_g = Lasso(alpha=0.1)

    model = PLR(data, score=score)
    model.set_learners(ml_l=ml_l, ml_m=ml_m, ml_g=ml_g)
    model.draw_sample_splitting(n_folds=N_FOLDS, n_rep=n_rep)
    model.fit()

    basis = pd.DataFrame(
        np.random.normal(0, 1, size=(N_OBS, BASIS_DIM)),
        columns=[f"b{i}" for i in range(BASIS_DIM)],
    )
    return model, basis


@pytest.fixture(scope="module", params=["partialling out", "IV-type"])
def score(request):
    return request.param


@pytest.fixture(scope="module")
def fitted_plr_single_rep(score) -> tuple[PLR, pd.DataFrame]:
    return _build_plr(n_rep=1, score=score)


@pytest.fixture(scope="module")
def fitted_plr_multi_rep(score) -> tuple[PLR, pd.DataFrame]:
    return _build_plr(n_rep=2, score=score)


@pytest.mark.ci
def test_cate_returns_blp(fitted_plr_single_rep):
    """cate() returns a fitted DoubleMLBLP instance."""
    model, basis = fitted_plr_single_rep
    cate = model.cate(basis)
    assert isinstance(cate, DoubleMLBLP)


@pytest.mark.ci
def test_cate_confint_shape(fitted_plr_single_rep):
    """cate().confint() returns a DataFrame with one row per basis column."""
    model, basis = fitted_plr_single_rep
    cate = model.cate(basis)
    ci = cate.confint()
    assert isinstance(ci, pd.DataFrame)
    assert ci.shape[0] == BASIS_DIM


@pytest.mark.ci
@pytest.mark.parametrize("cov_type", ["nonrobust", "HC1", "HC3"])
def test_cate_cov_type_passthrough(fitted_plr_single_rep, cov_type):
    """The cov_type kwarg propagates through to the underlying OLS fit."""
    model, basis = fitted_plr_single_rep
    cate = model.cate(basis, cov_type=cov_type)
    assert cate.blp_model[0].cov_type == cov_type


@pytest.mark.ci
def test_cate_multi_rep_n_rep(fitted_plr_multi_rep):
    """cate.n_rep matches the model's n_rep under multi-rep."""
    model, basis = fitted_plr_multi_rep
    cate = model.cate(basis)
    assert cate.n_rep == 2
    assert isinstance(cate.blp_model, list)
    assert len(cate.blp_model) == 2


@pytest.mark.ci
def test_cate_multi_rep_shapes(fitted_plr_multi_rep):
    """all_coef and all_se have shape (BASIS_DIM, n_rep) under multi-rep."""
    model, basis = fitted_plr_multi_rep
    cate = model.cate(basis)
    assert cate.all_coef.shape == (BASIS_DIM, 2)
    assert cate.all_se.shape == (BASIS_DIM, 2)
    assert isinstance(cate.confint(), pd.DataFrame)
    assert isinstance(cate.summary, pd.DataFrame)


@pytest.mark.ci
def test_gate_dummy_coded(fitted_plr_single_rep):
    """gate() accepts a pre-dummy-coded boolean DataFrame."""
    model, _ = fitted_plr_single_rep
    x1 = model._dml_data.x[:, 0]
    groups = pd.DataFrame({"low": x1 <= np.median(x1), "high": x1 > np.median(x1)})
    gate = model.gate(groups)
    assert isinstance(gate, DoubleMLBLP)
    assert all(gate.confint().index == groups.columns.to_list())


@pytest.mark.ci
def test_gate_single_column_string(fitted_plr_single_rep):
    """A single-column string DataFrame is auto-converted to dummies."""
    model, _ = fitted_plr_single_rep
    np.random.seed(0)
    groups = pd.DataFrame(np.random.choice(["A", "B"], N_OBS))
    gate = model.gate(groups)
    assert isinstance(gate, DoubleMLBLP)
    assert all(gate.confint().index == ["Group_A", "Group_B"])


@pytest.mark.ci
def test_gate_warns_small_group(fitted_plr_single_rep):
    """A group with <= 5 observations triggers a UserWarning."""
    model, _ = fitted_plr_single_rep
    groups = pd.DataFrame(
        {
            "small": np.array([True] * 3 + [False] * (N_OBS - 3)),
            "large": np.array([False] * 3 + [True] * (N_OBS - 3)),
        }
    )
    with pytest.warns(UserWarning, match=r"At least one group effect is estimated with less than 6 observations"):
        model.gate(groups)


@pytest.mark.ci
def test_cate_exception_before_fit():
    """Calling cate() before fit() raises ValueError."""
    np.random.seed(42)
    data = make_plr_CCDDHNR2018(n_obs=N_OBS, dim_x=5, alpha=0.5, return_type="DoubleMLData")
    model = PLR(data, score="partialling out")
    model.set_learners(ml_l=Lasso(alpha=0.1), ml_m=Lasso(alpha=0.1))
    basis = pd.DataFrame(np.random.normal(0, 1, size=(N_OBS, BASIS_DIM)))
    with pytest.raises(ValueError, match=r"requires a fitted model"):
        model.cate(basis)


@pytest.mark.ci
def test_gate_exception_not_dataframe(fitted_plr_single_rep):
    """gate() with a non-DataFrame raises TypeError."""
    model, _ = fitted_plr_single_rep
    with pytest.raises(TypeError, match=r"DataFrame type"):
        model.gate(np.zeros((N_OBS, 2)))


@pytest.mark.ci
def test_gate_exception_bad_dtype(fitted_plr_single_rep):
    """gate() with multi-column non-bool/int data raises TypeError."""
    model, _ = fitted_plr_single_rep
    groups = pd.DataFrame(
        {
            "g1": np.random.normal(0, 1, N_OBS),
            "g2": np.random.normal(0, 1, N_OBS),
        }
    )
    with pytest.raises(TypeError, match=r"bool type or int type"):
        model.gate(groups)


@pytest.mark.ci
@pytest.mark.parametrize("score", ["partialling out", "IV-type"])
@pytest.mark.parametrize("n_rep", [1, 2])
def test_cate_vs_legacy(score, n_rep):
    """CATE coefficients from the new PLR match the legacy DoubleMLPLR."""
    np.random.seed(42)
    data = make_plr_CCDDHNR2018(n_obs=N_OBS, dim_x=5, alpha=0.5, return_type="DoubleMLData")

    ml_l = Lasso(alpha=0.1)
    ml_m = Lasso(alpha=0.1)
    ml_g = Lasso(alpha=0.1)

    np.random.seed(3141)
    dml_old = dml.DoubleMLPLR(data, ml_l, ml_m, ml_g, n_folds=N_FOLDS, n_rep=n_rep, score=score)
    dml_old.fit()

    dml_new = PLR(data, score=score)
    dml_new.set_learners(ml_l=ml_l, ml_m=ml_m, ml_g=ml_g)
    dml_new._n_folds = N_FOLDS
    dml_new._n_rep = n_rep
    dml_new._smpls = dml_old.smpls
    dml_new.fit()

    np.random.seed(0)
    basis = pd.DataFrame(
        np.random.normal(0, 1, size=(N_OBS, BASIS_DIM)),
        columns=[f"b{i}" for i in range(BASIS_DIM)],
    )

    cate_old = dml_old.cate(basis)
    cate_new = dml_new.cate(basis)

    np.testing.assert_allclose(cate_new.coef, cate_old.coef, rtol=1e-9)
    np.testing.assert_allclose(cate_new.se, cate_old.se, rtol=1e-9)
