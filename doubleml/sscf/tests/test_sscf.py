import numpy as np
import pytest
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression

from doubleml.sscf import DoubleMLSSCF, generalized_inverse_mills_ratio, make_green_silence_data

from ._utils_sscf_manual import fit_sscf_manual


@pytest.fixture(scope="module", params=[True, False])
def nested(request):
    return request.param


@pytest.fixture(scope="module", params=[3, 5])
def n_folds(request):
    return request.param


@pytest.fixture(scope="module")
def dml_fixture(nested, n_folds):
    np.random.seed(3141)
    sim = make_green_silence_data(
        n_obs=800,
        dim_x=6,
        dim_z=4,
        n_deciles=4,
        theta=-0.5,
        n_active_outcome=2,
        n_active_selection_x=2,
        n_active_selection_z=3,
        return_type="dict",
    )
    dml_data = sim["dml_data"]

    ml_pi = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=5000)
    ml_g = LinearRegression()
    ml_m = LinearRegression()

    dml_obj = DoubleMLSSCF(dml_data, ml_pi, ml_g, ml_m, n_folds=n_folds, nested_cross_fitting=nested)
    dml_obj.fit()

    manual = fit_sscf_manual(
        dml_data.y,
        dml_data.d,
        dml_data.x,
        dml_data.z,
        dml_obj.smpls[0],
        ml_pi,
        ml_g,
        ml_m,
        nested=nested,
    )
    return {"dml_obj": dml_obj, "manual": manual, "sim": sim}


def test_coef_against_manual(dml_fixture):
    assert np.allclose(dml_fixture["dml_obj"].coef[0], dml_fixture["manual"]["theta"], rtol=1e-9, atol=1e-10)


def test_se_against_manual(dml_fixture):
    assert np.allclose(dml_fixture["dml_obj"].se[0], dml_fixture["manual"]["se"], rtol=1e-9, atol=1e-10)


def test_score_elements_against_manual(dml_fixture):
    psi_a = dml_fixture["dml_obj"].psi_elements["psi_a"][:, 0, 0]
    psi_b = dml_fixture["dml_obj"].psi_elements["psi_b"][:, 0, 0]
    assert np.allclose(psi_a, dml_fixture["manual"]["psi_a"], rtol=1e-9, atol=1e-10)
    assert np.allclose(psi_b, dml_fixture["manual"]["psi_b"], rtol=1e-9, atol=1e-10)


def test_score_elements_zero_outside_selected_sample(dml_fixture):
    d = dml_fixture["dml_obj"]._dml_data.d
    psi_a = dml_fixture["dml_obj"].psi_elements["psi_a"][:, 0, 0]
    psi_b = dml_fixture["dml_obj"].psi_elements["psi_b"][:, 0, 0]
    assert np.all(psi_a[d == 0] == 0.0)
    assert np.all(psi_b[d == 0] == 0.0)


def test_zero_padding_yields_effective_sample_size(dml_fixture):
    """The variance of the base class equals the variance based on n = sum(D)."""
    assert np.allclose(dml_fixture["dml_obj"].se[0], dml_fixture["manual"]["se_selected"], rtol=1e-9, atol=1e-10)
    assert np.allclose(dml_fixture["dml_obj"].coef[0], dml_fixture["manual"]["theta_selected"], rtol=1e-9, atol=1e-10)


def test_nuisance_elements_against_manual(dml_fixture):
    elements = dml_fixture["dml_obj"].nuisance_elements
    for key in ["pi_hat", "h_hat", "g_hat", "m_hat"]:
        assert np.allclose(elements[key][:, 0], dml_fixture["manual"][key], rtol=1e-9, atol=1e-10)


def test_score_test_against_manual(dml_fixture):
    test = dml_fixture["dml_obj"].selection_bias_test()
    assert np.allclose(test["statistic"], dml_fixture["manual"]["score_stat"], rtol=1e-9, atol=1e-10)
    assert 0.0 <= test["p_value"] <= 1.0
    assert test["n_selected"] == int(dml_fixture["dml_obj"]._dml_data.d.sum())


def test_bias_quantification(dml_fixture):
    res = dml_fixture["dml_obj"].bias_quantification()
    d = dml_fixture["dml_obj"]._dml_data.d
    y = dml_fixture["dml_obj"]._dml_data.y
    theta = dml_fixture["dml_obj"].coef[0]
    assert np.allclose(res["bias"], theta * res["imr"])
    assert np.allclose(res["y_corrected"][d == 1], (y - theta * res["imr"])[d == 1])
    assert np.all(np.isnan(res["y_corrected"][d == 0]))
    # unobservables that increase disclosure lower the outcome: theta < 0
    assert res["summary"]["mean_imr_non_selected"] > res["summary"]["mean_imr_selected"]


def test_generalized_imr_matches_probit_index():
    index = np.linspace(-2.0, 2.0, 101)
    from scipy.stats import norm

    propensity = norm.cdf(index)
    expected = norm.pdf(index) / norm.cdf(index)
    assert np.allclose(generalized_inverse_mills_ratio(propensity), expected, rtol=1e-8)


def test_consistency_under_no_selection_bias():
    """With theta_0 = 0 the estimator is centered at zero and the test keeps its size."""
    estimates, rejections = [], []
    for seed in range(20):
        np.random.seed(1000 + seed)
        dml_data = make_green_silence_data(n_obs=2000, dim_x=8, dim_z=5, n_deciles=5, theta=0.0)
        dml_obj = DoubleMLSSCF(
            dml_data,
            LogisticRegression(penalty="l1", C=0.1, solver="liblinear", max_iter=5000),
            LassoCV(max_iter=20000),
            LassoCV(max_iter=20000),
            n_folds=3,
        )
        dml_obj.fit()
        estimates.append(dml_obj.coef[0])
        rejections.append(dml_obj.selection_bias_test()["reject"])
    estimates = np.array(estimates)
    # the mean estimate is within three Monte Carlo standard errors of zero
    assert abs(estimates.mean()) < 3 * estimates.std(ddof=1) / np.sqrt(len(estimates))
    # the empirical size of the 5% test is not grossly distorted
    assert np.mean(rejections) < 0.30


def test_repeated_sample_splitting_runs():
    np.random.seed(42)
    dml_data = make_green_silence_data(
        n_obs=600, dim_x=5, dim_z=3, n_deciles=4, n_active_outcome=2, n_active_selection_x=2, n_active_selection_z=2
    )
    dml_obj = DoubleMLSSCF(
        dml_data, LogisticRegression(max_iter=5000), LinearRegression(), LinearRegression(), n_folds=3, n_rep=3
    )
    dml_obj.fit()
    assert dml_obj.all_coef.shape == (1, 3)
    assert dml_obj.nuisance_elements["h_hat"].shape == (600, 3)
    assert np.isfinite(dml_obj.selection_bias_test()["statistic"])


def test_bootstrap_and_confint_run():
    np.random.seed(7)
    dml_data = make_green_silence_data(
        n_obs=600, dim_x=5, dim_z=3, n_deciles=4, n_active_outcome=2, n_active_selection_x=2, n_active_selection_z=2
    )
    dml_obj = DoubleMLSSCF(dml_data, LogisticRegression(max_iter=5000), LinearRegression(), LinearRegression(), n_folds=3)
    dml_obj.fit()
    dml_obj.bootstrap(n_rep_boot=100)
    ci = dml_obj.confint(joint=True)
    assert ci.shape == (1, 2)
    assert ci.values[0, 0] < ci.values[0, 1]


def test_nonlinear_index_dgp():
    """The DGP supports a non-linear selection index f_0(Z), for which a linear
    probit/logistic learner is misspecified. The inverse Mills ratio identity
    h = phi(f_0)/Phi(f_0) = phi(Phi^{-1}(pi))/pi holds for any f_0."""
    from scipy.stats import norm

    np.random.seed(5)
    sim = make_green_silence_data(
        n_obs=1500,
        dim_x=6,
        dim_z=4,
        n_deciles=4,
        n_active_outcome=2,
        n_active_selection_x=2,
        n_active_selection_z=3,
        nonlinear_index=True,
        return_type="dict",
    )
    assert sim["nonlinear_index"]

    # the true propensity implied by the index, mapped back through the probit link;
    # restricted to the region where the Phi -> Phi^{-1} round trip is numerically stable
    propensity = norm.cdf(sim["index"])
    stable = (propensity > 1e-4) & (propensity < 1 - 1e-4)
    assert np.allclose(generalized_inverse_mills_ratio(propensity[stable]), sim["imr"][stable], atol=1e-6)

    # a linear index cannot reproduce f_0 (in contrast to the default DGP)
    z_full = np.column_stack((sim["x"], sim["u"]))
    r2_nonlinear = LinearRegression().fit(z_full, sim["index"]).score(z_full, sim["index"])
    assert r2_nonlinear < 0.8

    np.random.seed(5)
    sim_lin = make_green_silence_data(
        n_obs=1500,
        dim_x=6,
        dim_z=4,
        n_deciles=4,
        n_active_outcome=2,
        n_active_selection_x=2,
        n_active_selection_z=3,
        return_type="dict",
    )
    z_lin = np.column_stack((sim_lin["x"], sim_lin["u"]))
    assert LinearRegression().fit(z_lin, sim_lin["index"]).score(z_lin, sim_lin["index"]) > 0.999


def test_fit_runs_with_nonlinear_index():
    """The class fits without error when the selection index is non-linear."""
    np.random.seed(6)
    dml_data = make_green_silence_data(
        n_obs=1200,
        dim_x=6,
        dim_z=4,
        n_deciles=4,
        n_active_outcome=2,
        n_active_selection_x=2,
        n_active_selection_z=3,
        nonlinear_index=True,
    )
    dml_obj = DoubleMLSSCF(
        dml_data,
        LogisticRegression(max_iter=5000),
        LinearRegression(),
        LinearRegression(),
        n_folds=3,
    )
    dml_obj.fit()
    assert np.isfinite(dml_obj.coef[0]) and np.isfinite(dml_obj.se[0])
