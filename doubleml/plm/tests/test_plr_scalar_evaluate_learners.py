"""Tests for evaluate_learners(), nuisance_loss, and nuisance_targets on PLR scalar models."""

import numpy as np
import pytest
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from doubleml.plm.datasets import make_plr_CCDDHNR2018
from doubleml.plm.plr_scalar import PLR

N_OBS = 500
N_FOLDS = 5
N_REP = 2


@pytest.fixture(scope="module")
def plr_data():
    """Shared PLR dataset."""
    np.random.seed(3141)
    return make_plr_CCDDHNR2018(n_obs=N_OBS, dim_x=5)


@pytest.fixture(scope="module", params=["partialling out", "IV-type"])
def score(request):
    """Parametrize over PLR score functions."""
    return request.param


@pytest.fixture(scope="module")
def fitted_plr(score, plr_data):
    """Fit a PLR model for the given score."""
    model = PLR(plr_data, score=score)
    model.set_learners(ml_l=Lasso(), ml_m=Lasso())
    model.fit(n_folds=N_FOLDS, n_rep=N_REP)
    return model


# ==================== nuisance_loss ====================


@pytest.mark.ci
def test_nuisance_loss_type_and_shape(fitted_plr):
    """nuisance_loss is a dict of (n_rep,) arrays; ml_l/ml_m are finite; ml_g is NaN."""
    loss = fitted_plr.nuisance_loss

    assert isinstance(loss, dict)
    for name in ["ml_l", "ml_m"]:
        assert isinstance(loss[name], np.ndarray)
        assert loss[name].shape == (N_REP,)
        assert np.all(np.isfinite(loss[name]))

    if fitted_plr.score == "IV-type":
        assert isinstance(loss["ml_g"], np.ndarray)
        assert loss["ml_g"].shape == (N_REP,)
        assert np.all(np.isnan(loss["ml_g"]))


@pytest.mark.ci
def test_nuisance_loss_positive(fitted_plr):
    """RMSE values for ml_l and ml_m are strictly positive."""
    loss = fitted_plr.nuisance_loss
    assert np.all(loss["ml_l"] > 0)
    assert np.all(loss["ml_m"] > 0)


# ==================== nuisance_targets ====================


@pytest.mark.ci
def test_nuisance_targets_type_and_shape(fitted_plr):
    """nuisance_targets is a dict; ml_l/ml_m have real values; ml_g is all-NaN (IV-type)."""
    targets = fitted_plr.nuisance_targets

    assert isinstance(targets, dict)
    for name in ["ml_l", "ml_m"]:
        assert isinstance(targets[name], np.ndarray)
        assert targets[name].shape == (N_OBS, N_REP)
        assert not np.all(np.isnan(targets[name]))

    if fitted_plr.score == "IV-type":
        assert isinstance(targets["ml_g"], np.ndarray)
        assert targets["ml_g"].shape == (N_OBS, N_REP)
        assert np.all(np.isnan(targets["ml_g"]))


@pytest.mark.ci
def test_nuisance_targets_ml_l_equals_y(fitted_plr, plr_data):
    """ml_l target is y broadcast across repetitions."""
    targets = fitted_plr.nuisance_targets
    y = plr_data.y
    for i_rep in range(N_REP):
        np.testing.assert_array_equal(targets["ml_l"][:, i_rep], y)


@pytest.mark.ci
def test_nuisance_targets_ml_m_equals_d(fitted_plr, plr_data):
    """ml_m target is d broadcast across repetitions."""
    targets = fitted_plr.nuisance_targets
    d = plr_data.d
    for i_rep in range(N_REP):
        np.testing.assert_array_equal(targets["ml_m"][:, i_rep], d)


# ==================== evaluate_learners ====================


@pytest.mark.ci
def test_evaluate_learners_default(fitted_plr):
    """Default evaluate_learners() returns RMSE for ml_l and ml_m."""
    result = fitted_plr.evaluate_learners()

    assert isinstance(result, dict)
    for name in ["ml_l", "ml_m"]:
        assert isinstance(result[name], np.ndarray)
        assert result[name].shape == (N_REP,)
        assert np.all(result[name] > 0)


@pytest.mark.ci
def test_evaluate_learners_rmse_matches_nuisance_loss(fitted_plr):
    """evaluate_learners with root_mean_squared_error matches nuisance_loss for ml_l and ml_m."""
    result = fitted_plr.evaluate_learners(metric=root_mean_squared_error)
    loss = fitted_plr.nuisance_loss

    np.testing.assert_allclose(result["ml_l"], loss["ml_l"], rtol=1e-9)
    np.testing.assert_allclose(result["ml_m"], loss["ml_m"], rtol=1e-9)


@pytest.mark.ci
def test_evaluate_learners_r2(fitted_plr):
    """evaluate_learners with r2_score returns values <= 1 with correct shape."""
    result = fitted_plr.evaluate_learners(learners=["ml_l", "ml_m"], metric=r2_score)

    for name in ["ml_l", "ml_m"]:
        assert result[name].shape == (N_REP,)
        assert np.all(result[name] <= 1.0)


@pytest.mark.ci
def test_evaluate_learners_mae(fitted_plr):
    """evaluate_learners with mean_absolute_error returns positive values with correct shape."""
    result = fitted_plr.evaluate_learners(learners=["ml_l", "ml_m"], metric=mean_absolute_error)

    for name in ["ml_l", "ml_m"]:
        assert result[name].shape == (N_REP,)
        assert np.all(result[name] > 0)


@pytest.mark.ci
def test_evaluate_learners_subset(fitted_plr):
    """Requesting only ml_l returns only the ml_l key."""
    result = fitted_plr.evaluate_learners(learners=["ml_l"])

    assert list(result.keys()) == ["ml_l"]
    assert result["ml_l"].shape == (N_REP,)


@pytest.mark.ci
def test_evaluate_learners_custom_metric(fitted_plr):
    """A custom lambda metric produces consistent results."""
    custom_mae = lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))  # noqa: E731
    result_custom = fitted_plr.evaluate_learners(learners=["ml_l"], metric=custom_mae)
    result_sklearn = fitted_plr.evaluate_learners(learners=["ml_l"], metric=mean_absolute_error)

    np.testing.assert_allclose(result_custom["ml_l"], result_sklearn["ml_l"], rtol=1e-9)


# ==================== Before-fit errors ====================


@pytest.mark.ci
def test_evaluate_learners_before_fit_raises(plr_data):
    """evaluate_learners() raises ValueError before fit_nuisance_models()."""
    model = PLR(plr_data)
    model.set_learners(ml_l=Lasso(), ml_m=Lasso())

    msg = r"Call fit\(\) or fit_nuisance_models\(\) first"
    with pytest.raises(ValueError, match=msg):
        model.evaluate_learners()


@pytest.mark.ci
def test_evaluate_learners_after_reset_raises(plr_data):
    """evaluate_learners() raises ValueError after draw_sample_splitting() resets fit state."""
    model = PLR(plr_data)
    model.set_learners(ml_l=Lasso(), ml_m=Lasso())
    model.draw_sample_splitting(n_folds=N_FOLDS, n_rep=N_REP)
    model.fit_nuisance_models()
    # Re-drawing splits resets fit state
    model.draw_sample_splitting(n_folds=N_FOLDS, n_rep=N_REP)

    msg = r"Call fit\(\) or fit_nuisance_models\(\) first"
    with pytest.raises(ValueError, match=msg):
        model.evaluate_learners()


@pytest.mark.ci
def test_nuisance_loss_before_fit_raises(plr_data):
    """nuisance_loss raises ValueError before fit_nuisance_models()."""
    model = PLR(plr_data)
    model.set_learners(ml_l=Lasso(), ml_m=Lasso())

    msg = r"Call fit\(\) or fit_nuisance_models\(\) first"
    with pytest.raises(ValueError, match=msg):
        _ = model.nuisance_loss


@pytest.mark.ci
def test_nuisance_targets_before_fit_raises(plr_data):
    """nuisance_targets raises ValueError before fit_nuisance_models()."""
    model = PLR(plr_data)
    model.set_learners(ml_l=Lasso(), ml_m=Lasso())

    msg = r"Call fit\(\) or fit_nuisance_models\(\) first"
    with pytest.raises(ValueError, match=msg):
        _ = model.nuisance_targets


# ==================== Input validation ====================


@pytest.mark.ci
def test_evaluate_learners_invalid_learner(fitted_plr):
    """Requesting an unknown learner name raises ValueError."""
    with pytest.raises(ValueError, match=r"Invalid learner"):
        fitted_plr.evaluate_learners(learners=["ml_l", "ml_unknown"])


@pytest.mark.ci
def test_evaluate_learners_invalid_metric(fitted_plr):
    """Passing a non-callable metric raises TypeError."""
    with pytest.raises(TypeError, match=r"metric must be callable"):
        fitted_plr.evaluate_learners(metric="rmse")


# ==================== Reset behaviour ====================


@pytest.mark.ci
def test_reset_clears_nuisance(plr_data):
    """After draw_sample_splitting(), nuisance_loss raises ValueError."""
    model = PLR(plr_data)
    model.set_learners(ml_l=Lasso(), ml_m=Lasso())
    model.fit(n_folds=N_FOLDS, n_rep=N_REP)
    assert model.nuisance_loss is not None

    model.draw_sample_splitting(n_folds=N_FOLDS, n_rep=N_REP)

    msg = r"Call fit\(\) or fit_nuisance_models\(\) first"
    with pytest.raises(ValueError, match=msg):
        _ = model.nuisance_loss
