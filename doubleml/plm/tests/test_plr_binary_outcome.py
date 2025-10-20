import math

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import doubleml as dml

from ...tests._utils import draw_smpls
from ._utils_plr_manual import boot_plr, fit_plr, fit_sensitivity_elements_plr


@pytest.fixture(
    scope="module", params=[RandomForestClassifier(max_depth=2, n_estimators=10), LogisticRegression(max_iter=1000)]
)
def learner_binary(request):
    return request.param


@pytest.fixture(scope="module", params=["partialling out"])
def score(request):
    return request.param


@pytest.fixture(scope="module")
def generate_binary_data():
    """Generate synthetic data with binary outcome"""
    np.random.seed(42)
    n = 500
    p = 5

    # Generate covariates
    X = np.random.normal(0, 1, size=(n, p))

    # Generate treatment
    d_prob = 1 / (1 + np.exp(-(X[:, 0] + X[:, 1] + np.random.normal(0, 1, n))))
    d = np.random.binomial(1, d_prob)

    # Generate binary outcome with treatment effect
    theta_true = 0.5  # true treatment effect
    y_prob = 1 / (1 + np.exp(-(X[:, 0] + X[:, 2] + theta_true * d + np.random.normal(0, 0.5, n))))
    y = np.random.binomial(1, y_prob)

    # Combine into DataFrame
    data = pd.DataFrame({"y": y, "d": d, **{f"X{i+1}": X[:, i] for i in range(p)}})

    return data


@pytest.mark.ci
def test_dml_plr_binary_warnings(generate_binary_data, learner_binary, score):
    data = generate_binary_data
    obj_dml_data = dml.DoubleMLData(data, "y", ["d"])
    msg = "The ml_l learner .+ was identified as classifier. Fitting an additive probability model."
    with pytest.warns(UserWarning, match=msg):
        _ = dml.DoubleMLPLR(obj_dml_data, clone(learner_binary), clone(learner_binary), score=score)


@pytest.mark.ci
def test_dml_plr_binary_exceptions(generate_binary_data, learner_binary, score):
    data = generate_binary_data
    obj_dml_data = dml.DoubleMLData(data, "X1", ["d"])
    msg = "The ml_l learner .+ was identified as classifier but the outcome variable is not binary with values 0 and 1."
    with pytest.raises(ValueError, match=msg):
        _ = dml.DoubleMLPLR(obj_dml_data, clone(learner_binary), clone(learner_binary), score=score)

    # IV-type not possible with binary outcome
    obj_dml_data = dml.DoubleMLData(data, "y", ["d"])
    msg = r"For score = 'IV-type', additive probability models \(binary outcomes\) are not supported."
    with pytest.raises(ValueError, match=msg):
        _ = dml.DoubleMLPLR(obj_dml_data, clone(learner_binary), clone(learner_binary), score="IV-type")


@pytest.fixture(scope="module")
def dml_plr_binary_fixture(generate_binary_data, learner_binary, score):
    boot_methods = ["normal"]
    n_folds = 2
    n_rep_boot = 502

    # collect data
    data = generate_binary_data
    x_cols = data.columns[data.columns.str.startswith("X")].tolist()

    # Set machine learning methods for m & g
    ml_l = clone(learner_binary)
    ml_m = clone(learner_binary)

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData(data, "y", ["d"], x_cols)
    dml_plr_obj = dml.DoubleMLPLR(obj_dml_data, ml_l, ml_m, n_folds=n_folds, score=score)
    dml_plr_obj.fit()

    np.random.seed(3141)
    y = data["y"].values
    x = data.loc[:, x_cols].values
    d = data["d"].values
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds)

    res_manual = fit_plr(y, x, d, clone(learner_binary), clone(learner_binary), clone(learner_binary), all_smpls, score)

    np.random.seed(3141)
    # test with external nuisance predictions
    dml_plr_obj_ext = dml.DoubleMLPLR(obj_dml_data, ml_l, ml_m, n_folds, score=score)

    # synchronize the sample splitting
    dml_plr_obj_ext.set_sample_splitting(all_smpls=all_smpls)
    prediction_dict = {
        "d": {
            "ml_l": dml_plr_obj.predictions["ml_l"].reshape(-1, 1),
            "ml_m": dml_plr_obj.predictions["ml_m"].reshape(-1, 1),
        }
    }
    dml_plr_obj_ext.fit(external_predictions=prediction_dict)

    res_dict = {
        "coef": dml_plr_obj.coef.item(),
        "coef_manual": res_manual["theta"],
        "coef_ext": dml_plr_obj_ext.coef.item(),
        "se": dml_plr_obj.se.item(),
        "se_manual": res_manual["se"],
        "se_ext": dml_plr_obj_ext.se.item(),
        "boot_methods": boot_methods,
    }

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_t_stat = boot_plr(
            y,
            d,
            res_manual["thetas"],
            res_manual["ses"],
            res_manual["all_l_hat"],
            res_manual["all_m_hat"],
            res_manual["all_g_hat"],
            all_smpls,
            score,
            bootstrap,
            n_rep_boot,
        )

        np.random.seed(3141)
        dml_plr_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        np.random.seed(3141)
        dml_plr_obj_ext.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict["boot_t_stat" + bootstrap] = dml_plr_obj.boot_t_stat
        res_dict["boot_t_stat" + bootstrap + "_manual"] = boot_t_stat.reshape(-1, 1, 1)
        res_dict["boot_t_stat" + bootstrap + "_ext"] = dml_plr_obj_ext.boot_t_stat

    # sensitivity tests
    res_dict["sensitivity_elements"] = dml_plr_obj.sensitivity_elements
    res_dict["sensitivity_elements_manual"] = fit_sensitivity_elements_plr(
        y, d.reshape(-1, 1), all_coef=dml_plr_obj.all_coef, predictions=dml_plr_obj.predictions, score=score, n_rep=1
    )
    # check if sensitivity score with rho=0 gives equal asymptotic standard deviation
    dml_plr_obj.sensitivity_analysis(rho=0.0)
    res_dict["sensitivity_ses"] = dml_plr_obj.sensitivity_params["se"]

    return res_dict


@pytest.mark.ci
def test_dml_plr_binary_coef(dml_plr_binary_fixture):
    assert math.isclose(dml_plr_binary_fixture["coef"], dml_plr_binary_fixture["coef_manual"], rel_tol=1e-9, abs_tol=1e-4)
    assert math.isclose(dml_plr_binary_fixture["coef"], dml_plr_binary_fixture["coef_ext"], rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_plr_binary_se(dml_plr_binary_fixture):
    assert math.isclose(dml_plr_binary_fixture["se"], dml_plr_binary_fixture["se_manual"], rel_tol=1e-9, abs_tol=1e-4)
    assert math.isclose(dml_plr_binary_fixture["se"], dml_plr_binary_fixture["se_ext"], rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_plr_binary_boot(dml_plr_binary_fixture):
    for bootstrap in dml_plr_binary_fixture["boot_methods"]:
        assert np.allclose(
            dml_plr_binary_fixture["boot_t_stat" + bootstrap],
            dml_plr_binary_fixture["boot_t_stat" + bootstrap + "_manual"],
            rtol=1e-9,
            atol=1e-4,
        )
        assert np.allclose(
            dml_plr_binary_fixture["boot_t_stat" + bootstrap],
            dml_plr_binary_fixture["boot_t_stat" + bootstrap + "_ext"],
            rtol=1e-9,
            atol=1e-4,
        )


@pytest.mark.ci
def test_dml_plr_binary_sensitivity(dml_plr_binary_fixture):
    sensitivity_element_names = ["sigma2", "nu2", "psi_sigma2", "psi_nu2"]
    for sensitivity_element in sensitivity_element_names:
        assert np.allclose(
            dml_plr_binary_fixture["sensitivity_elements"][sensitivity_element],
            dml_plr_binary_fixture["sensitivity_elements_manual"][sensitivity_element],
        )


@pytest.mark.ci
def test_dml_plr_binary_sensitivity_rho0(dml_plr_binary_fixture):
    assert np.allclose(dml_plr_binary_fixture["se"], dml_plr_binary_fixture["sensitivity_ses"]["lower"], rtol=1e-9, atol=1e-4)
    assert np.allclose(dml_plr_binary_fixture["se"], dml_plr_binary_fixture["sensitivity_ses"]["upper"], rtol=1e-9, atol=1e-4)


@pytest.fixture(scope="module", params=["nonrobust", "HC0", "HC1", "HC2", "HC3"])
def cov_type(request):
    return request.param


@pytest.mark.ci
def test_dml_plr_binary_cate_gate(score, cov_type, generate_binary_data):
    n = 12

    # Use generated binary data
    data = generate_binary_data.head(n)
    x_cols = data.columns[data.columns.str.startswith("X")].tolist()

    obj_dml_data = dml.DoubleMLData(data, "y", ["d"], x_cols)
    ml_l = LogisticRegression(max_iter=1000)
    ml_m = LogisticRegression(max_iter=1000)

    dml_plr_obj = dml.DoubleMLPLR(obj_dml_data, ml_l, ml_m, n_folds=2, score=score)
    dml_plr_obj.fit()

    random_basis = pd.DataFrame(np.random.normal(0, 1, size=(n, 3)))
    cate = dml_plr_obj.cate(random_basis, cov_type=cov_type)
    assert isinstance(cate, dml.DoubleMLBLP)
    assert isinstance(cate.confint(), pd.DataFrame)
    assert cate.blp_model.cov_type == cov_type

    groups_1 = pd.DataFrame(np.column_stack([data["X1"] <= 0, data["X1"] > 0.2]), columns=["Group 1", "Group 2"])
    msg = "At least one group effect is estimated with less than 6 observations."
    with pytest.warns(UserWarning, match=msg):
        gate_1 = dml_plr_obj.gate(groups_1, cov_type=cov_type)
    assert isinstance(gate_1, dml.utils.blp.DoubleMLBLP)
    assert isinstance(gate_1.confint(), pd.DataFrame)
    assert all(gate_1.confint().index == groups_1.columns.tolist())
    assert gate_1.blp_model.cov_type == cov_type
