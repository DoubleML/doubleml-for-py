"""Tests for evaluate_learners(), nuisance_loss, and nuisance_targets on IRM scalar models."""

import numpy as np
import pytest
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import log_loss, mean_absolute_error, r2_score, root_mean_squared_error

from doubleml.irm.datasets import make_irm_data
from doubleml.irm.irm_scalar import IRM

N_OBS = 500
N_FOLDS = 5
N_REP = 2


@pytest.fixture(scope="module")
def irm_data():
    """Shared IRM dataset."""
    np.random.seed(3141)
    return make_irm_data(n_obs=N_OBS, dim_x=5)


@pytest.fixture(scope="module", params=["ATE", "ATTE"])
def score(request):
    """Parametrize over IRM score functions."""
    return request.param


@pytest.fixture(scope="module")
def fitted_irm(score, irm_data):
    """Fit an IRM model for the given score."""
    model = IRM(irm_data, score=score)
    model.set_learners(ml_g=Lasso(), ml_m=LogisticRegression())
    model.fit(n_folds=N_FOLDS, n_rep=N_REP)
    return model


# ==================== nuisance_loss ====================


@pytest.mark.ci
def test_nuisance_loss_type_and_shape(fitted_irm):
    """nuisance_loss is a dict of (n_rep,) arrays; all entries are finite."""
    loss = fitted_irm.nuisance_loss

    assert isinstance(loss, dict)
    for name in ["ml_g0", "ml_g1", "ml_m"]:
        assert isinstance(loss[name], np.ndarray)
        assert loss[name].shape == (N_REP,)
        assert np.all(np.isfinite(loss[name]))


@pytest.mark.ci
def test_nuisance_loss_ml_m_is_logloss(fitted_irm):
    """ml_m loss uses logloss (classifier path) — positive finite values."""
    loss = fitted_irm.nuisance_loss
    assert np.all(loss["ml_m"] > 0)


@pytest.mark.ci
def test_nuisance_loss_ml_g_is_rmse(fitted_irm):
    """ml_g0 and ml_g1 loss uses RMSE (regressor path) — positive finite values."""
    loss = fitted_irm.nuisance_loss
    assert np.all(loss["ml_g0"] > 0)
    assert np.all(loss["ml_g1"] > 0)


# ==================== nuisance_targets ====================


@pytest.mark.ci
def test_nuisance_targets_type_and_shape(fitted_irm):
    """nuisance_targets is a dict; all entries are (n_obs, n_rep) arrays."""
    targets = fitted_irm.nuisance_targets

    assert isinstance(targets, dict)
    for name in ["ml_g0", "ml_g1", "ml_m"]:
        assert isinstance(targets[name], np.ndarray)
        assert targets[name].shape == (N_OBS, N_REP)


@pytest.mark.ci
def test_nuisance_targets_ml_g0_partial_nan(fitted_irm, irm_data):
    """ml_g0 target is y where d==0 and NaN where d==1."""
    targets = fitted_irm.nuisance_targets
    d = irm_data.d

    for i_rep in range(N_REP):
        col = targets["ml_g0"][:, i_rep]
        assert np.all(np.isnan(col[d == 1]))
        assert np.all(np.isfinite(col[d == 0]))


@pytest.mark.ci
def test_nuisance_targets_ml_g1_partial_nan(fitted_irm, irm_data):
    """ml_g1 target is y where d==1 and NaN where d==0."""
    targets = fitted_irm.nuisance_targets
    d = irm_data.d

    for i_rep in range(N_REP):
        col = targets["ml_g1"][:, i_rep]
        assert np.all(np.isnan(col[d == 0]))
        assert np.all(np.isfinite(col[d == 1]))


@pytest.mark.ci
def test_nuisance_targets_ml_m_equals_d(fitted_irm, irm_data):
    """ml_m target is d broadcast across repetitions."""
    targets = fitted_irm.nuisance_targets
    d = irm_data.d
    for i_rep in range(N_REP):
        np.testing.assert_array_equal(targets["ml_m"][:, i_rep], d)


# ==================== evaluate_learners ====================


@pytest.mark.ci
def test_evaluate_learners_default(fitted_irm):
    """Default evaluate_learners() returns finite values with correct shape."""
    result = fitted_irm.evaluate_learners()

    assert isinstance(result, dict)
    for name in ["ml_g0", "ml_g1", "ml_m"]:
        assert isinstance(result[name], np.ndarray)
        assert result[name].shape == (N_REP,)
        assert np.all(np.isfinite(result[name]))


@pytest.mark.ci
def test_evaluate_learners_logloss_ml_m_matches_nuisance_loss(fitted_irm):
    """evaluate_learners with log_loss on ml_m matches nuisance_loss['ml_m']."""
    result = fitted_irm.evaluate_learners(learners=["ml_m"], metric=log_loss)
    loss = fitted_irm.nuisance_loss

    np.testing.assert_allclose(result["ml_m"], loss["ml_m"], rtol=1e-9)


@pytest.mark.ci
def test_evaluate_learners_rmse_ml_g_matches_nuisance_loss(fitted_irm):
    """evaluate_learners with RMSE on ml_g0/g1 matches nuisance_loss."""
    result = fitted_irm.evaluate_learners(learners=["ml_g0", "ml_g1"], metric=root_mean_squared_error)
    loss = fitted_irm.nuisance_loss

    np.testing.assert_allclose(result["ml_g0"], loss["ml_g0"], rtol=1e-9)
    np.testing.assert_allclose(result["ml_g1"], loss["ml_g1"], rtol=1e-9)


@pytest.mark.ci
def test_evaluate_learners_partial_nans_ml_g(fitted_irm):
    """RMSE for ml_g0/g1 is finite despite NaN targets for the opposite treatment group."""
    result = fitted_irm.evaluate_learners(learners=["ml_g0", "ml_g1"], metric=root_mean_squared_error)

    assert np.all(np.isfinite(result["ml_g0"]))
    assert np.all(np.isfinite(result["ml_g1"]))


@pytest.mark.ci
def test_evaluate_learners_r2(fitted_irm):
    """evaluate_learners with r2_score returns values <= 1 with correct shape."""
    result = fitted_irm.evaluate_learners(learners=["ml_g0", "ml_g1"], metric=r2_score)

    for name in ["ml_g0", "ml_g1"]:
        assert result[name].shape == (N_REP,)
        assert np.all(result[name] <= 1.0)


@pytest.mark.ci
def test_evaluate_learners_mae(fitted_irm):
    """evaluate_learners with mean_absolute_error returns positive values."""
    result = fitted_irm.evaluate_learners(learners=["ml_g0", "ml_g1"], metric=mean_absolute_error)

    for name in ["ml_g0", "ml_g1"]:
        assert result[name].shape == (N_REP,)
        assert np.all(result[name] > 0)


@pytest.mark.ci
def test_evaluate_learners_subset(fitted_irm):
    """Requesting only ml_m returns only the ml_m key."""
    result = fitted_irm.evaluate_learners(learners=["ml_m"])

    assert list(result.keys()) == ["ml_m"]
    assert result["ml_m"].shape == (N_REP,)


@pytest.mark.ci
def test_evaluate_learners_custom_metric(fitted_irm):
    """A custom lambda metric produces consistent results."""
    custom_mae = lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))  # noqa: E731
    result_custom = fitted_irm.evaluate_learners(learners=["ml_g0"], metric=custom_mae)
    result_sklearn = fitted_irm.evaluate_learners(learners=["ml_g0"], metric=mean_absolute_error)

    np.testing.assert_allclose(result_custom["ml_g0"], result_sklearn["ml_g0"], rtol=1e-9)


# ==================== Before-fit errors ====================


@pytest.mark.ci
def test_evaluate_learners_before_fit_raises(irm_data):
    """evaluate_learners() raises ValueError before fit()."""
    model = IRM(irm_data)
    model.set_learners(ml_g=Lasso(), ml_m=LogisticRegression())

    msg = r"Call fit\(\) or fit_nuisance_models\(\) first"
    with pytest.raises(ValueError, match=msg):
        model.evaluate_learners()


@pytest.mark.ci
def test_nuisance_loss_before_fit_raises(irm_data):
    """nuisance_loss raises ValueError before fit()."""
    model = IRM(irm_data)
    model.set_learners(ml_g=Lasso(), ml_m=LogisticRegression())

    msg = r"Call fit\(\) or fit_nuisance_models\(\) first"
    with pytest.raises(ValueError, match=msg):
        _ = model.nuisance_loss


@pytest.mark.ci
def test_nuisance_targets_before_fit_raises(irm_data):
    """nuisance_targets raises ValueError before fit()."""
    model = IRM(irm_data)
    model.set_learners(ml_g=Lasso(), ml_m=LogisticRegression())

    msg = r"Call fit\(\) or fit_nuisance_models\(\) first"
    with pytest.raises(ValueError, match=msg):
        _ = model.nuisance_targets


# ==================== Input validation ====================


@pytest.mark.ci
def test_evaluate_learners_invalid_learner(fitted_irm):
    """Requesting an unknown learner name raises ValueError."""
    with pytest.raises(ValueError, match=r"Invalid learner"):
        fitted_irm.evaluate_learners(learners=["ml_g0", "ml_unknown"])


@pytest.mark.ci
def test_evaluate_learners_invalid_metric(fitted_irm):
    """Passing a non-callable metric raises TypeError."""
    with pytest.raises(TypeError, match=r"metric must be callable"):
        fitted_irm.evaluate_learners(metric="rmse")


# ==================== Reset behaviour ====================


@pytest.mark.ci
def test_reset_clears_nuisance(irm_data):
    """After draw_sample_splitting(), nuisance_loss raises ValueError."""
    model = IRM(irm_data)
    model.set_learners(ml_g=Lasso(), ml_m=LogisticRegression())
    model.fit(n_folds=N_FOLDS, n_rep=N_REP)
    assert model.nuisance_loss is not None

    model.draw_sample_splitting(n_folds=N_FOLDS, n_rep=N_REP)

    msg = r"Call fit\(\) or fit_nuisance_models\(\) first"
    with pytest.raises(ValueError, match=msg):
        _ = model.nuisance_loss
