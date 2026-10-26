import warnings

import numpy as np
import plotly.graph_objects as go
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml.double_ml_framework import concat
from doubleml.irm.irm import DoubleMLIRM


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module")
def dml_framework_sensitivity_fixture(n_rep, generate_data_simple):
    dml_data, dml_data_2 = generate_data_simple

    ml_g = LinearRegression()
    ml_m = LogisticRegression()

    dml_irm_obj = DoubleMLIRM(dml_data, ml_g, ml_m, n_rep=n_rep)
    dml_irm_obj.fit()
    dml_framework_obj = dml_irm_obj.construct_framework()
    dml_framework_obj.sensitivity_analysis()

    # add objects
    dml_framework_obj_add_obj = dml_framework_obj + dml_framework_obj
    dml_framework_obj_add_obj.sensitivity_analysis()

    # substract objects
    dml_irm_obj_2 = DoubleMLIRM(dml_data_2, ml_g, ml_m, n_rep=n_rep)
    dml_irm_obj_2.fit()
    dml_framework_obj_2 = dml_irm_obj_2.construct_framework()
    dml_framework_obj_sub_obj = dml_framework_obj - dml_framework_obj_2
    dml_framework_obj_sub_obj2 = dml_framework_obj - dml_framework_obj
    dml_framework_obj_sub_obj2.sensitivity_analysis()

    # multiply objects
    dml_framework_obj_mul_obj = dml_framework_obj * 2
    dml_framework_obj_mul_obj.sensitivity_analysis()
    dml_framework_obj_mul_zero_obj = 0 * dml_framework_obj
    dml_framework_obj_mul_zero_obj.sensitivity_analysis()

    # concat objects
    dml_framework_obj_concat = concat([dml_framework_obj, dml_framework_obj])

    result_dict = {
        "dml_obj": dml_irm_obj,
        "dml_obj_2": dml_irm_obj_2,
        "dml_framework_obj": dml_framework_obj,
        "dml_framework_obj_2": dml_framework_obj_2,
        "dml_framework_obj_add_obj": dml_framework_obj_add_obj,
        "dml_framework_obj_sub_obj": dml_framework_obj_sub_obj,
        "dml_framework_obj_sub_obj2": dml_framework_obj_sub_obj2,
        "dml_framework_obj_mul_obj": dml_framework_obj_mul_obj,
        "dml_framework_obj_mul_zero_obj": dml_framework_obj_mul_zero_obj,
        "dml_framework_obj_concat": dml_framework_obj_concat,
        "n_rep": n_rep,
    }
    return result_dict


@pytest.mark.ci
def test_dml_framework_sensitivity_shapes(dml_framework_sensitivity_fixture):
    n_rep = dml_framework_sensitivity_fixture["dml_framework_obj"].n_rep
    n_thetas = dml_framework_sensitivity_fixture["dml_framework_obj"].n_thetas
    n_obs = dml_framework_sensitivity_fixture["dml_framework_obj"].n_obs

    object_list = [
        "dml_framework_obj",
        "dml_framework_obj_2",
        "dml_framework_obj_add_obj",
        "dml_framework_obj_sub_obj",
        "dml_framework_obj_mul_obj",
    ]
    var_keys = ["max_bias"]
    score_keys = ["psi_max_bias"]

    for obj in object_list:
        assert dml_framework_sensitivity_fixture[obj]._sensitivity_implemented
        for key in var_keys:
            assert dml_framework_sensitivity_fixture[obj].sensitivity_elements[key].shape == (1, n_thetas, n_rep)
        for key in score_keys:
            assert dml_framework_sensitivity_fixture[obj].sensitivity_elements[key].shape == (n_obs, n_thetas, n_rep)

    # separate test for concat
    for key in var_keys:
        assert dml_framework_sensitivity_fixture["dml_framework_obj_concat"].sensitivity_elements[key].shape == (1, 2, n_rep)
    for key in score_keys:
        assert dml_framework_sensitivity_fixture["dml_framework_obj_concat"].sensitivity_elements[key].shape == (
            n_obs,
            2,
            n_rep,
        )


@pytest.mark.ci
def test_dml_framework_sensitivity_summary(dml_framework_sensitivity_fixture):
    # summary without sensitivity analysis
    sensitivity_summary = dml_framework_sensitivity_fixture["dml_framework_obj_2"].sensitivity_summary
    substring = "Apply sensitivity_analysis() to generate sensitivity_summary."
    assert substring in sensitivity_summary

    # summary with sensitivity analysis
    sensitivity_summary = dml_framework_sensitivity_fixture["dml_framework_obj"].sensitivity_summary
    assert isinstance(sensitivity_summary, str)
    substrings = [
        "\n------------------ Scenario          ------------------\n",
        "\n------------------ Bounds with CI    ------------------\n",
        "\n------------------ Robustness Values ------------------\n",
        "Significance Level: level=",
        "Sensitivity parameters: cf_y=",
    ]
    for substring in substrings:
        assert substring in sensitivity_summary


@pytest.mark.ci
def test_dml_framework_sensitivity_operations(dml_framework_sensitivity_fixture):
    add_obj = dml_framework_sensitivity_fixture["dml_framework_obj_add_obj"]
    mul_obj = dml_framework_sensitivity_fixture["dml_framework_obj_mul_obj"]
    mul_zero_obj = dml_framework_sensitivity_fixture["dml_framework_obj_mul_zero_obj"]

    for key in ["theta", "se", "ci"]:
        assert np.allclose(add_obj.sensitivity_params[key]["upper"], mul_obj.sensitivity_params[key]["upper"])
        assert np.allclose(add_obj.sensitivity_params[key]["lower"], mul_obj.sensitivity_params[key]["lower"])

        assert mul_zero_obj.sensitivity_params[key]["upper"] == 0
        assert mul_zero_obj.sensitivity_params[key]["lower"] == 0

    assert np.allclose(add_obj.sensitivity_params["rv"], mul_obj.sensitivity_params["rv"])
    assert mul_zero_obj.sensitivity_params["rv"] > 0.99  # due to degenerated variance
    assert mul_zero_obj.sensitivity_params["rva"] > 0.99  # due to degenerated variance

    sub_obj = dml_framework_sensitivity_fixture["dml_framework_obj_sub_obj2"]
    assert np.allclose(sub_obj.sensitivity_params["theta"]["upper"], -1 * sub_obj.sensitivity_params["theta"]["lower"])
    assert np.allclose(sub_obj.sensitivity_params["se"]["upper"], sub_obj.sensitivity_params["se"]["lower"])
    assert np.allclose(sub_obj.sensitivity_params["ci"]["upper"], -1 * sub_obj.sensitivity_params["ci"]["lower"])
    assert sub_obj.sensitivity_params["rv"] < 0.01
    assert sub_obj.sensitivity_params["rva"] < 0.01


@pytest.mark.ci
def test_dml_framework_sensitivity_plot_orientation(dml_framework_sensitivity_fixture):
    """Contour value z[i, j] equals the bound at cf_d=x[j], cf_y=y[i] for asymmetric grid bounds."""
    framework = dml_framework_sensitivity_fixture["dml_framework_obj"]
    fig = framework.sensitivity_plot(grid_bounds=(0.1, 0.3), grid_size=11)
    contour = fig.data[0]
    z = np.asarray(contour.z)
    assert z.shape == (len(contour.y), len(contour.x))

    null_hypothesis = framework.sensitivity_params["input"]["null_hypothesis"][0]
    bound = "upper" if null_hypothesis > framework.thetas[0] else "lower"
    for i, j in [(10, 2), (2, 10), (5, 7)]:
        expected = framework._calc_sensitivity_analysis(cf_y=contour.y[i], cf_d=contour.x[j], rho=1.0, level=0.95)
        np.testing.assert_allclose(z[i, j], expected["theta"][bound][0], rtol=1e-12)


@pytest.mark.ci
def test_dml_framework_sensitivity_plot_markers_match_contour(dml_framework_sensitivity_fixture):
    """Scenario and benchmarks are evaluated with the rho of the plot, with a warning if it differs from the scenario."""
    framework = dml_framework_sensitivity_fixture["dml_framework_obj"]
    cf_d_vec = np.linspace(0, 0.1, 11)
    cf_y_vec = np.linspace(0, 0.3, 11)
    benchmarks = {"cf_y": [cf_y_vec[9]], "cf_d": [cf_d_vec[4]], "name": ["bench"]}

    msg = r"The scenario is evaluated with the values of sensitivity_plot\(\)."
    with pytest.warns(UserWarning, match=msg):
        fig = framework.sensitivity_plot(rho=0.5, benchmarks=benchmarks, grid_bounds=(0.1, 0.3), grid_size=11)
    z = np.asarray(fig.data[0].z)
    benchmark_trace = next(trace for trace in fig.data if trace.name == "Benchmarks")
    np.testing.assert_allclose(benchmark_trace.customdata[0][0], z[9, 4], rtol=1e-12)

    null_hypothesis = framework.sensitivity_params["input"]["null_hypothesis"][0]
    bound = "upper" if null_hypothesis > framework.thetas[0] else "lower"
    scenario_input = framework.sensitivity_params["input"]
    expected_scenario = framework._calc_sensitivity_analysis(
        cf_y=scenario_input["cf_y"], cf_d=scenario_input["cf_d"], rho=0.5, level=0.95
    )["theta"][bound][0]
    scenario_trace = next(trace for trace in fig.data if trace.name == "Scenario")
    assert f": {round(expected_scenario, 3)}</b>" in scenario_trace.hovertemplate

    # no warning if rho and level match the scenario
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _ = framework.sensitivity_plot(grid_size=10)


@pytest.mark.ci
def test_dml_framework_sensitivity_plot_null_hypothesis(dml_framework_sensitivity_fixture):
    """The null_hypothesis argument determines the bound and defaults to the value of sensitivity_analysis()."""
    framework = dml_framework_sensitivity_fixture["dml_framework_obj"]
    theta = framework.thetas[0]
    z_upper = np.asarray(framework.sensitivity_plot(null_hypothesis=float(theta + 1.0), grid_size=10).data[0].z)
    z_lower = np.asarray(framework.sensitivity_plot(null_hypothesis=float(theta - 1.0), grid_size=10).data[0].z)
    z_default = np.asarray(framework.sensitivity_plot(grid_size=10).data[0].z)

    assert np.all(z_upper >= theta)
    assert np.all(z_lower <= theta)
    assert np.any(z_upper > z_lower)
    expected_default = z_upper if framework.sensitivity_params["input"]["null_hypothesis"][0] > theta else z_lower
    np.testing.assert_allclose(z_default, expected_default)


@pytest.mark.ci
def test_dml_framework_sensitivity_plot_defaults_from_analysis(dml_framework_sensitivity_fixture):
    """rho and level default to the values of sensitivity_analysis(), so the scenario matches sensitivity_params."""
    framework = dml_framework_sensitivity_fixture["dml_obj"].construct_framework()
    framework.sensitivity_analysis(rho=0.5, level=0.9)

    fig_default = framework.sensitivity_plot(value="ci", grid_size=10)
    fig_explicit = framework.sensitivity_plot(value="ci", rho=0.5, level=0.9, grid_size=10)
    np.testing.assert_allclose(np.asarray(fig_default.data[0].z), np.asarray(fig_explicit.data[0].z))

    null_hypothesis = framework.sensitivity_params["input"]["null_hypothesis"][0]
    bound = "upper" if null_hypothesis > framework.thetas[0] else "lower"
    expected_scenario = framework.sensitivity_params["ci"][bound][0]
    scenario_trace = next(trace for trace in fig_default.data if trace.name == "Scenario")
    assert f": {round(expected_scenario, 3)}</b>" in scenario_trace.hovertemplate

    msg = r"The scenario is evaluated with the values of sensitivity_plot\(\)."
    with pytest.warns(UserWarning, match=msg):
        _ = framework.sensitivity_plot(rho=1.0, grid_size=10)


@pytest.mark.ci
def test_dml_framework_sensitivity_plot_without_analysis(dml_framework_sensitivity_fixture):
    """sensitivity_plot works without sensitivity_analysis() if no scenario is included."""
    framework = dml_framework_sensitivity_fixture["dml_obj_2"].construct_framework()
    assert framework.sensitivity_params is None
    for value in ["theta", "ci"]:
        fig = framework.sensitivity_plot(value=value, include_scenario=False, grid_size=10)
        assert isinstance(fig, go.Figure)
        assert all(trace.name != "Scenario" for trace in fig.data)
