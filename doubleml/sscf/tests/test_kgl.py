import numpy as np
import pytest

from doubleml.sscf import (
    AdaptiveKernelGroupLasso,
    adaptive_weights,
    kg_lasso_path,
    lambda_max,
    make_green_silence_data,
    post_selection_refit,
    selection_metrics,
    variable_selection,
)
from doubleml.sscf.kgl import _prox_kernel_group


def test_prox_reduces_to_group_soft_thresholding():
    """For K = I the operator is the standard group lasso soft-thresholding rule."""
    rng = np.random.default_rng(42)
    identity = np.eye(5)
    for _ in range(20):
        v = rng.normal(size=5)
        tau = rng.uniform(0.0, 2.0)
        expected = max(1.0 - tau / np.linalg.norm(v), 0.0) * v
        assert np.allclose(_prox_kernel_group(v, identity, identity, tau), expected, atol=1e-10)


def test_prox_exact_solves_the_proximal_problem():
    """The exact operator attains the minimum of the proximal objective."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(0)
    dim = 6
    root = rng.normal(size=(dim, dim))
    k_mat = root @ root.T / dim + 0.1 * np.eye(dim)
    k_inv = np.linalg.inv(k_mat)
    for _ in range(5):
        v = rng.normal(size=dim)
        tau = rng.uniform(0.05, 0.5)

        def objective(b):
            return 0.5 * np.sum((b - v) ** 2) + tau * np.sqrt(b @ k_mat @ b)

        b_exact = _prox_kernel_group(v, k_mat, k_inv, tau, method="exact")
        b_numeric = minimize(
            objective, np.zeros(dim), method="Nelder-Mead", options=dict(maxiter=200000, xatol=1e-10, fatol=1e-12)
        ).x
        assert objective(b_exact) <= objective(b_numeric) + 1e-8


def test_prox_variants_coincide_for_identity_kernel():
    """The closed form of the paper is exact when K is proportional to the identity."""
    rng = np.random.default_rng(1)
    dim = 4
    for scale in [0.5, 1.0, 2.0]:
        k_mat = scale * np.eye(dim)
        k_inv = np.linalg.inv(k_mat)
        for _ in range(10):
            v = rng.normal(size=dim)
            tau = rng.uniform(0.01, 1.0)
            b_exact = _prox_kernel_group(v, k_mat, k_inv, tau, method="exact")
            b_paper = _prox_kernel_group(v, k_mat, k_inv, tau, method="paper")
            assert np.allclose(b_exact, b_paper, atol=1e-8)


def test_prox_returns_zero_below_dual_norm():
    k_mat = np.array([[1.0, 0.4], [0.4, 1.0]]) / 2
    k_inv = np.linalg.inv(k_mat)
    v = np.array([0.05, -0.02])
    tau = 10.0
    assert np.allclose(_prox_kernel_group(v, k_mat, k_inv, tau), 0.0)


@pytest.fixture(scope="module")
def toy_design():
    np.random.seed(123)
    sim = make_green_silence_data(
        n_obs=1500,
        dim_x=8,
        dim_z=4,
        n_deciles=5,
        n_active_outcome=2,
        n_active_selection_x=2,
        n_active_selection_z=3,
        theta=-0.5,
        return_type="dict",
    )
    return sim


def test_lambda_max_yields_empty_model(toy_design):
    sim = toy_design
    sel = sim["d"] == 1
    k, y = sim["k"][sel], sim["y"][sel]
    lam = lambda_max(k, y, sim["groups"], sim["K_mats"])
    model = AdaptiveKernelGroupLasso(lam=lam * 1.001).fit(k, y, sim["groups"], sim["K_mats"])
    assert model.active_groups_.size == 0
    model_below = AdaptiveKernelGroupLasso(lam=lam * 0.9).fit(k, y, sim["groups"], sim["K_mats"])
    assert model_below.active_groups_.size > 0


def test_objective_decreases_monotonically(toy_design):
    sim = toy_design
    sel = sim["d"] == 1
    k, y, groups, k_mats = sim["k"][sel], sim["y"][sel], sim["groups"], sim["K_mats"]
    lam = 0.3 * lambda_max(k, y, groups, k_mats)

    objectives = []
    b_warm = None
    for max_iter in [1, 2, 5, 20, 100]:
        model = AdaptiveKernelGroupLasso(lam=lam, max_iter=max_iter, tol=0.0).fit(k, y, groups, k_mats, b_init=b_warm)
        objectives.append(model.objective_)
    assert all(objectives[i + 1] <= objectives[i] + 1e-8 for i in range(len(objectives) - 1))


def test_group_structure_of_solution(toy_design):
    """Selection happens at the group level: coefficients are zero group-wise."""
    sim = toy_design
    sel = sim["d"] == 1
    k, y, groups, k_mats = sim["k"][sel], sim["y"][sel], sim["groups"], sim["K_mats"]
    lam = 0.5 * lambda_max(k, y, groups, k_mats)
    model = AdaptiveKernelGroupLasso(lam=lam).fit(k, y, groups, k_mats)
    for j in np.unique(groups):
        block = model.coef_[groups == j]
        assert np.all(block == 0.0) or np.all(block != 0.0) or np.any(block != 0.0)
        if j not in model.active_groups_:
            assert np.allclose(block, 0.0)


def test_adaptive_weights_penalize_selection_signal():
    groups = np.repeat(np.arange(3), 4)
    b_init = np.concatenate([np.ones(4), np.ones(4), np.zeros(4)])
    beta_hat = np.array([0.0, 2.0, 0.0])
    w = adaptive_weights(beta_hat, b_init, groups, gamma=2.0)
    # a variable with a strong selection signal receives a larger weight
    assert w[1] > w[0]
    # a group with a zero initial estimate is effectively excluded
    assert w[2] > 1e6


def test_adaptive_weights_wrong_length():
    groups = np.repeat(np.arange(3), 2)
    with pytest.raises(ValueError, match="one entry per group"):
        adaptive_weights(np.array([0.0, 1.0]), np.ones(6), groups)


def test_post_selection_refit_matches_ols(toy_design):
    sim = toy_design
    sel = sim["d"] == 1
    k, y, groups = sim["k"][sel], sim["y"][sel], sim["groups"]
    active = np.array([0, 1])
    res = post_selection_refit(k, y, groups, active)
    cols = np.where(np.isin(groups, active))[0]
    design = np.column_stack((np.ones(k.shape[0]), k[:, cols]))
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    assert np.allclose(res["intercept"], coef[0])
    assert np.allclose(res["coef"][cols], coef[1:])
    assert np.allclose(res["coef"][~np.isin(np.arange(k.shape[1]), cols)], 0.0)


def test_selection_metrics():
    met = selection_metrics([0, 1, 5], [0, 1, 2], n_groups=10)
    assert met["tp"] == 2 and met["fp"] == 1 and met["fn"] == 1
    assert met["tpr"] == pytest.approx(2 / 3)
    assert met["fpr"] == pytest.approx(1 / 7)
    assert not met["exact_recovery"]
    met_exact = selection_metrics([0, 1], [0, 1], n_groups=5)
    assert met_exact["exact_recovery"]
    assert met_exact["tpr"] == 1.0 and met_exact["fpr"] == 0.0


def test_path_selects_intermediate_lambda(toy_design):
    sim = toy_design
    sel = sim["d"] == 1
    res = kg_lasso_path(sim["k"][sel], sim["y"][sel], sim["groups"], sim["K_mats"], n_lambda=15)
    assert 0 < res["index"] < 14
    assert res["ebic"].shape == (15,)
    assert res["lambda"] == pytest.approx(res["lambdas"][res["index"]])


def test_variable_selection_recovers_active_groups(toy_design):
    """With the true theta and IMR, the corrected KG-lasso recovers the active set."""
    sim = toy_design
    sel = sim["d"] == 1
    beta_hat = np.zeros(sim["K_mats"].__len__())
    beta_hat[sim["active_selection_x"]] = 1.0

    res = variable_selection(
        sim["y"][sel],
        sim["k"][sel],
        sim["groups"],
        sim["K_mats"],
        beta_hat,
        theta_hat=sim["theta"],
        imr=sim["imr"][sel],
        n_lambda=20,
    )
    met = selection_metrics(res["active_groups"], sim["active_outcome"], len(sim["K_mats"]))
    assert met["tpr"] == 1.0
    assert met["fpr"] < 0.5


def test_variable_selection_requires_imr(toy_design):
    sim = toy_design
    sel = sim["d"] == 1
    with pytest.raises(ValueError, match="imr"):
        variable_selection(sim["y"][sel], sim["k"][sel], sim["groups"], sim["K_mats"], np.zeros(8), theta_hat=-0.5, imr=None)
