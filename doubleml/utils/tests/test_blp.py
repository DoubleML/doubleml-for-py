import copy

import numpy as np
import pandas as pd
import pytest

import doubleml as dml

from ._utils_blp_manual import blp_confint, fit_blp


@pytest.fixture(scope="module", params=[True, False])
def ci_joint(request):
    return request.param


@pytest.fixture(scope="module", params=[0.95, 0.9])
def ci_level(request):
    return request.param


@pytest.fixture(scope="module", params=["nonrobust", "HC0", "HC1", "HC2", "HC3"])
def cov_type(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def use_t(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module")
def dml_blp_fixture(ci_joint, ci_level, cov_type, use_t, n_rep):
    n = 50
    kwargs = {"cov_type": cov_type, "use_t": use_t}

    np.random.seed(42)
    random_basis = pd.DataFrame(np.random.normal(0, 1, size=(n, 3)))
    random_signal = np.random.normal(0, 1, size=(n, n_rep))

    blp = dml.DoubleMLBLP(random_signal, random_basis)

    blp_obj = copy.copy(blp)
    blp.fit(**kwargs)
    blp_manual = []
    for i in range(n_rep):
        blp_manual.append(fit_blp(random_signal[:, i], random_basis, **kwargs))

    np.random.seed(42)
    ci_1 = blp.confint(random_basis, joint=ci_joint, level=ci_level, n_rep_boot=1000)
    np.random.seed(42)
    ci_2 = blp.confint(joint=ci_joint, level=ci_level, n_rep_boot=1000)
    expected_ci_2 = []
    for i in range(n_rep):
        expected_ci_2.append(
            np.vstack(
                (
                    blp.blp_model[i].conf_int(alpha=(1 - ci_level))[0],
                    blp.blp_model[i].params,
                    blp.blp_model[i].conf_int(alpha=(1 - ci_level))[1],
                )
            ).T
        )
    expected_ci_2 = np.median(np.array(expected_ci_2), axis=0)

    np.random.seed(42)
    ci_manual = []
    for i in range(n_rep):
        ci_manual.append(blp_confint(blp_manual[i], random_basis, joint=ci_joint, level=ci_level, n_rep_boot=1000))
    ci_manual = np.median(np.array([ci_manual[i].to_numpy() for i in range(n_rep)]), axis=0)

    coef_manual = np.median(np.array([blp_manual[i].params for i in range(n_rep)]), axis=0)
    omega_manual = np.transpose(np.array([blp_manual[i].cov_params().to_numpy() for i in range(n_rep)]), (1, 2, 0))

    fittedvalues_manual = np.array([blp_manual[i].fittedvalues for i in range(n_rep)])
    fittedvalues = np.array([blp.blp_model[i].fittedvalues for i in range(n_rep)])

    res_dict = {
        "coef": blp.coef,
        "coef_manual": coef_manual,
        "values": fittedvalues,
        "values_manual": fittedvalues_manual,
        "omega": blp.blp_omega,
        "omega_manual": omega_manual,
        "basis": blp.basis,
        "signal": blp.orth_signal,
        "ci_1": ci_1,
        "ci_2": ci_2,
        "expected_ci_2": expected_ci_2,
        "ci_manual": ci_manual,
        "blp_model": blp,
        "unfitted_blp_model": blp_obj,
    }

    return res_dict


@pytest.mark.ci
def test_dml_blp_coef(dml_blp_fixture):
    assert np.allclose(dml_blp_fixture["coef"], dml_blp_fixture["coef_manual"], rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_blp_values(dml_blp_fixture):
    assert np.allclose(dml_blp_fixture["values"], dml_blp_fixture["values_manual"], rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_blp_omega(dml_blp_fixture):
    assert np.allclose(dml_blp_fixture["omega"], dml_blp_fixture["omega_manual"], rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_blp_ci_2(dml_blp_fixture):
    assert np.allclose(dml_blp_fixture["expected_ci_2"], dml_blp_fixture["ci_2"], rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_blp_ci_1(dml_blp_fixture):
    assert np.allclose(dml_blp_fixture["ci_1"], dml_blp_fixture["ci_manual"], rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_blp_return_types(dml_blp_fixture):
    assert isinstance(dml_blp_fixture["blp_model"].__str__(), str)
    assert isinstance(dml_blp_fixture["blp_model"].summary, pd.DataFrame)
    assert isinstance(dml_blp_fixture["unfitted_blp_model"].summary, pd.DataFrame)


@pytest.mark.ci
def test_dml_blp_defaults():
    n = 50
    np.random.seed(42)
    random_basis = pd.DataFrame(np.random.normal(0, 1, size=(n, 3)))
    random_signal = np.random.normal(0, 1, size=(n,))

    blp = dml.DoubleMLBLP(random_signal, random_basis)
    assert blp.blp_omega is None
    blp.fit()

    assert np.allclose(blp.blp_omega[:, :, 0], blp.blp_model[0].cov_HC0, rtol=1e-9, atol=1e-4)

    assert blp._is_gate is False


@pytest.mark.ci
def test_doubleml_exception_blp():
    random_basis = pd.DataFrame(np.random.normal(0, 1, size=(2, 3)))
    signal = np.array([1, 2])
    signal_mismatch = np.array([1, 2, 3])

    msg = "The signal must be of np.ndarray type. Signal of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml.DoubleMLBLP(orth_signal=1, basis=random_basis)
    msg = "The signal must be one- or two-dimensional. Signal of dimensions 3 was passed."
    with pytest.raises(ValueError, match=msg):
        dml.DoubleMLBLP(orth_signal=np.array([[[1]], [[2]]]), basis=random_basis)
    msg = "The basis must be of DataFrame type or a list of DataFrames. Basis of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml.DoubleMLBLP(orth_signal=signal, basis=1)
    msg = "The number of observations in signal and basis does not match. Got 3 and 2."
    with pytest.raises(ValueError, match=msg):
        dml.DoubleMLBLP(orth_signal=signal_mismatch, basis=random_basis)
    msg = "Invalid pd.DataFrame: Contains duplicate column names."
    with pytest.raises(ValueError, match=msg):
        dml.DoubleMLBLP(orth_signal=signal, basis=pd.DataFrame(np.array([[1, 2], [4, 5]]), columns=["a_1", "a_1"]))

    dml_blp_confint = dml.DoubleMLBLP(orth_signal=signal, basis=random_basis)
    msg = r"Apply fit\(\) before confint\(\)."
    with pytest.raises(ValueError, match=msg):
        dml_blp_confint.confint(random_basis)

    dml_blp_confint.fit()
    msg = "joint must be True or False. Got 1."
    with pytest.raises(TypeError, match=msg):
        dml_blp_confint.confint(random_basis, joint=1)
    msg = "The confidence level must be of float type. 5% of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_blp_confint.confint(random_basis, level="5%")
    msg = r"The confidence level must be in \(0,1\). 0.0 was passed."
    with pytest.raises(ValueError, match=msg):
        dml_blp_confint.confint(random_basis, level=0.0)
    msg = "The number of bootstrap replications must be of int type. 500 of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_blp_confint.confint(random_basis, n_rep_boot="500")
    msg = "The number of bootstrap replications must be positive. 0 was passed."
    with pytest.raises(ValueError, match=msg):
        dml_blp_confint.confint(random_basis, n_rep_boot=0)
    msg = "The basis must be of DataFrame type. Basis of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_blp_confint.confint(basis=1)
    msg = "Invalid basis: DataFrame has to have the exact same number and ordering of columns."
    with pytest.raises(ValueError, match=msg):
        dml_blp_confint.confint(basis=pd.DataFrame(np.array([[1], [4]]), columns=["a_1"]))
    msg = "Invalid basis: DataFrame has to have the exact same number and ordering of columns."
    with pytest.raises(ValueError, match=msg):
        dml_blp_confint.confint(basis=pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]), columns=["x_1", "x_2", "x_3"]))


@pytest.mark.ci
def test_blp_per_rep_basis_fits():
    """A list-of-DataFrames basis fits and exposes per-rep coefficient shapes."""
    n, d, n_rep = 50, 3, 3
    np.random.seed(0)
    signal = np.random.normal(0, 1, size=(n, n_rep))
    cols = [f"b{i}" for i in range(d)]
    basis_list = [pd.DataFrame(np.random.normal(0, 1, size=(n, d)), columns=cols) for _ in range(n_rep)]

    blp = dml.DoubleMLBLP(signal, basis_list).fit()
    assert blp.all_coef.shape == (d, n_rep)
    assert blp.all_se.shape == (d, n_rep)
    assert blp.coef.shape == (d,)
    assert blp.se.shape == (d,)


@pytest.mark.ci
def test_blp_per_rep_basis_matches_shared():
    """Per-rep list of identical bases yields the same fit as the shared-basis call."""
    n, d, n_rep = 50, 3, 3
    np.random.seed(1)
    signal = np.random.normal(0, 1, size=(n, n_rep))
    basis = pd.DataFrame(np.random.normal(0, 1, size=(n, d)), columns=[f"b{i}" for i in range(d)])

    blp_shared = dml.DoubleMLBLP(signal, basis).fit()
    blp_list = dml.DoubleMLBLP(signal, [basis] * n_rep).fit()

    np.testing.assert_allclose(blp_list.all_coef, blp_shared.all_coef, rtol=1e-12)
    np.testing.assert_allclose(blp_list.all_se, blp_shared.all_se, rtol=1e-12)
    np.testing.assert_allclose(blp_list.coef, blp_shared.coef, rtol=1e-12)


@pytest.mark.ci
def test_blp_per_rep_basis_wrong_length():
    """Wrong list length raises ValueError."""
    n, n_rep = 30, 3
    signal = np.zeros((n, n_rep))
    basis = pd.DataFrame(np.zeros((n, 2)), columns=["a", "b"])
    with pytest.raises(ValueError, match=r"length n_rep=3"):
        dml.DoubleMLBLP(signal, [basis, basis])


@pytest.mark.ci
def test_blp_per_rep_basis_mismatched_columns():
    """Per-rep bases with different column names raise ValueError."""
    n, n_rep = 30, 2
    signal = np.zeros((n, n_rep))
    basis_a = pd.DataFrame(np.zeros((n, 2)), columns=["a", "b"])
    basis_b = pd.DataFrame(np.zeros((n, 2)), columns=["a", "c"])
    with pytest.raises(ValueError, match=r"same column names"):
        dml.DoubleMLBLP(signal, [basis_a, basis_b])


@pytest.mark.ci
def test_blp_per_rep_basis_mismatched_n_obs():
    """Per-rep basis with wrong row count raises ValueError."""
    n, n_rep = 30, 2
    signal = np.zeros((n, n_rep))
    basis_ok = pd.DataFrame(np.zeros((n, 2)), columns=["a", "b"])
    basis_bad = pd.DataFrame(np.zeros((n - 1, 2)), columns=["a", "b"])
    with pytest.raises(ValueError, match=r"basis entry 1"):
        dml.DoubleMLBLP(signal, [basis_ok, basis_bad])


@pytest.mark.ci
def test_blp_per_rep_basis_non_dataframe_entry():
    """A non-DataFrame entry in the list raises TypeError."""
    n, n_rep = 30, 2
    signal = np.zeros((n, n_rep))
    basis = pd.DataFrame(np.zeros((n, 2)), columns=["a", "b"])
    with pytest.raises(TypeError, match=r"All entries of basis list must be of DataFrame type"):
        dml.DoubleMLBLP(signal, [basis, np.zeros((n, 2))])
