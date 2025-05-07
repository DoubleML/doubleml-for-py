import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml.data import DoubleMLPanelData
from doubleml.did import DoubleMLDIDMulti
from doubleml.did.datasets import make_did_CS2021


@pytest.fixture(scope="module", params=["observational", "experimental"])
def did_score(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module")
def doubleml_did_fixture(did_score, n_rep):
    n_obs = 1000
    dgp = 5  # has to be experimental (for experimental score to be valid)
    np.random.seed(42)
    df = make_did_CS2021(n_obs=n_obs, dgp=dgp, n_pre_treat_periods=3, n_periods=5, time_type="float")
    dml_data = DoubleMLPanelData(df, y_col="y", d_cols="d", t_col="t", id_col="id", x_cols=["Z1", "Z2", "Z3", "Z4"])

    kwargs = {
        "obj_dml_data": dml_data,
        "ml_g": LinearRegression(),
        "ml_m": LogisticRegression(),
        "gt_combinations": "all",
        "score": did_score,
        "n_rep": n_rep,
        "n_folds": 2,
        "draw_sample_splitting": True,
    }

    dml_did = DoubleMLDIDMulti(**kwargs)

    np.random.seed(3141)
    dml_did.fit()

    res_dict = {
        "model": dml_did,
    }
    return res_dict


@pytest.mark.ci
def test_plot_bootstrap_warnings(doubleml_did_fixture):
    msg = "Joint confidence intervals require bootstrapping"
    with pytest.warns(UserWarning, match=msg):
        _ = doubleml_did_fixture["model"].plot_effects()


@pytest.mark.ci
def test_plot_effects_default(doubleml_did_fixture):
    dml_obj = doubleml_did_fixture["model"]
    fig, axes = dml_obj.plot_effects()

    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, list)
    assert all(isinstance(ax, plt.Axes) for ax in axes)
    plt.close("all")


@pytest.mark.ci
def test_plot_effects_confidence_level(doubleml_did_fixture):
    """Test plot_effects with different confidence levels."""
    dml_obj = doubleml_did_fixture["model"]

    # Test with 90% confidence level
    fig, _ = dml_obj.plot_effects(level=0.9)
    assert isinstance(fig, plt.Figure)

    # assert figure is not equal to default value
    fig_default, _ = dml_obj.plot_effects()
    assert fig_default != fig

    plt.close("all")


@pytest.mark.ci
def test_plot_effects_joint_ci(doubleml_did_fixture):
    """Test plot_effects with different joint confidence interval settings."""
    dml_obj = doubleml_did_fixture["model"]

    # Test with joint=False
    fig, _ = dml_obj.plot_effects(joint=False)
    assert isinstance(fig, plt.Figure)

    # assert figure is not equal to default value
    fig_default, _ = dml_obj.plot_effects()
    assert fig_default != fig

    plt.close("all")


@pytest.mark.ci
def test_plot_effects_figure_size(doubleml_did_fixture):
    """Test plot_effects with custom figure size."""
    dml_obj = doubleml_did_fixture["model"]

    custom_figsize = (10, 5)
    fig, _ = dml_obj.plot_effects(figsize=custom_figsize)
    assert isinstance(fig, plt.Figure)

    # Check if figure size matches the specified size
    width, height = fig.get_size_inches()
    assert (width, height) == custom_figsize

    plt.close("all")


@pytest.mark.ci
def test_plot_effects_color_palette(doubleml_did_fixture):
    """Test plot_effects with different color palettes."""
    dml_obj = doubleml_did_fixture["model"]

    # Test with a different seaborn palette
    fig, _ = dml_obj.plot_effects(color_palette="Set1")
    assert isinstance(fig, plt.Figure)

    # Test with a custom color list
    custom_colors = [(1, 0, 0), (0, 1, 0)]  # Red and green
    fig, _ = dml_obj.plot_effects(color_palette=custom_colors)
    assert isinstance(fig, plt.Figure)

    plt.close("all")


@pytest.mark.ci
def test_plot_effects_labels_and_title(doubleml_did_fixture):
    """Test plot_effects with custom labels and title."""
    dml_obj = doubleml_did_fixture["model"]

    custom_title = "Custom Title for Test"
    custom_ylabel = "Custom Y Label"

    fig, axes = dml_obj.plot_effects(title=custom_title, y_label=custom_ylabel)
    assert isinstance(fig, plt.Figure)

    # Check if title is set correctly (title is on the figure level)
    assert fig._suptitle.get_text() == custom_title

    # Check if y_label is set correctly (at least on the first axis)
    assert axes[0].get_ylabel() == custom_ylabel

    plt.close("all")


@pytest.mark.ci
def test_plot_effects_jitter(doubleml_did_fixture):
    """Test plot_effects with custom jitter settings."""
    dml_obj = doubleml_did_fixture["model"]

    # Test with custom jitter value
    fig, _ = dml_obj.plot_effects(jitter_value=0.2)
    assert isinstance(fig, plt.Figure)

    # assert figure is not equal to default value
    fig_default, _ = dml_obj.plot_effects()
    assert fig_default != fig

    # Test with custom default_jitter
    fig, _ = dml_obj.plot_effects(default_jitter=0.05)
    assert isinstance(fig, plt.Figure)

    # assert figure is not equal to default value
    fig_default, _ = dml_obj.plot_effects()
    assert fig_default != fig

    plt.close("all")
