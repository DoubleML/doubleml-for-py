import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml

df = dml.did.datasets.make_did_cs_CS2021(n_obs=500, dgp_type=1, n_pre_treat_periods=2, n_periods=4, time_type="float")
dml_data = dml.data.DoubleMLPanelData(df, y_col="y", d_cols="d", id_col="id", t_col="t", x_cols=["Z1", "Z2", "Z3", "Z4"])

# n_periods=5 dataset: t_values=[0,1,2,3,4], g_values={2.0, 3.0, 4.0, inf}
df5 = dml.did.datasets.make_did_cs_CS2021(n_obs=500, dgp_type=1, n_pre_treat_periods=2, n_periods=5, time_type="float")
dml_data5 = dml.data.DoubleMLPanelData(df5, y_col="y", d_cols="d", id_col="id", t_col="t", x_cols=["Z1", "Z2", "Z3", "Z4"])

args = {
    "obj_dml_data": dml_data,
    "ml_g": LinearRegression(),
    "ml_m": LogisticRegression(),
    "g_value": 2,
    "t_value_pre": 0,
    "t_value_eval": 1,
    "score": "observational",
    "n_rep": 1,
}

_base_args = {"ml_g": LinearRegression(), "ml_m": LogisticRegression(), "draw_sample_splitting": False}


def _make_model(dml_data_obj, g_val, t_pre, t_eval, control_group):
    return dml.did.DoubleMLDIDCSBinary(
        obj_dml_data=dml_data_obj,
        g_value=g_val,
        t_value_pre=t_pre,
        t_value_eval=t_eval,
        control_group=control_group,
        **_base_args,
    )


def _expected_n_obs_subset(data, g_col, t_col, t_values, g_val, t_pre, t_eval, control_group):
    """Compute expected n_obs_subset by replicating _preprocess_data subset logic.

    CS data is long-format (one row per unit per period), so this counts rows across
    both periods, not unique units (contrast with the DID binary helper which deduplicates
    by id_col). Assumes anticipation_periods=0 (the default); the +anticipation_periods
    offset in production code is not replicated here.
    """
    data_ts = data[data[t_col].isin([t_pre, t_eval])]
    G = data_ts[g_col] == g_val
    never_treated = np.isinf(data_ts[g_col])

    if control_group == "never_treated":
        C = never_treated
    else:
        comparison_period = max(t_eval, g_val)
        idx = np.where(t_values == comparison_period)[0][0]
        max_g_value = t_values[min(idx, len(t_values) - 1)]
        later_treated = (data_ts[g_col] > max_g_value) & ~G
        C = never_treated | later_treated

    return int((G | C).sum())


@pytest.mark.ci
def test_control_groups_different():
    dml_did_never_treated = dml.did.DoubleMLDIDCSBinary(control_group="never_treated", **args)
    dml_did_not_yet_treated = dml.did.DoubleMLDIDCSBinary(control_group="not_yet_treated", **args)

    assert dml_did_never_treated.n_obs_subset != dml_did_not_yet_treated.n_obs_subset
    # same treatment group
    assert dml_did_never_treated.data_subset["G_indicator"].sum() == dml_did_not_yet_treated.data_subset["G_indicator"].sum()

    dml_did_never_treated.fit()
    dml_did_not_yet_treated.fit()

    assert dml_did_never_treated.coef != dml_did_not_yet_treated.coef


@pytest.mark.ci
def test_never_treated_control_group_size():
    """n_obs_subset for never_treated equals (treatment group + never-treated) rows across both periods."""
    data, g_col, t_col = dml_data.data, dml_data.g_col, dml_data.t_col
    t_values = dml_data.t_values

    # pre-treatment: g_value=3 > t_value_eval=1
    g_val, t_pre, t_eval = 3.0, 0, 1
    expected = _expected_n_obs_subset(data, g_col, t_col, t_values, g_val, t_pre, t_eval, "never_treated")
    obj = _make_model(dml_data, g_val, t_pre, t_eval, "never_treated")
    assert obj.n_obs_subset == expected

    # post-treatment: g_value=2 <= t_value_eval=2
    g_val, t_pre, t_eval = 2.0, 0, 2
    expected = _expected_n_obs_subset(data, g_col, t_col, t_values, g_val, t_pre, t_eval, "never_treated")
    obj = _make_model(dml_data, g_val, t_pre, t_eval, "never_treated")
    assert obj.n_obs_subset == expected


@pytest.mark.ci
def test_not_yet_treated_control_group_size_post_treatment():
    """n_obs_subset for not_yet_treated in a post-treatment period includes later-treated units."""
    # Post-treatment: g_value=2, t_value_eval=2 → comparison_period=2, includes g=3 as later-treated
    g_val, t_pre, t_eval = 2.0, 0, 2
    data, g_col, t_col = dml_data.data, dml_data.g_col, dml_data.t_col
    t_values = dml_data.t_values

    expected = _expected_n_obs_subset(data, g_col, t_col, t_values, g_val, t_pre, t_eval, "not_yet_treated")
    obj = _make_model(dml_data, g_val, t_pre, t_eval, "not_yet_treated")
    assert obj.n_obs_subset == expected

    # later-treated units (g=3) are present in the control group
    control_g_vals = obj.data_subset.loc[obj.data_subset["C_indicator"] == 1, g_col].unique()
    assert 3.0 in control_g_vals


@pytest.mark.ci
def test_not_yet_treated_control_group_size_pre_treatment():
    """n_obs_subset for not_yet_treated in a pre-treatment period excludes intermediate groups."""
    # Pre-treatment: g_value=3 > t_value_eval=1 → comparison_period=3, excludes g=2 from control
    g_val, t_pre, t_eval = 3.0, 0, 1
    data, g_col, t_col = dml_data.data, dml_data.g_col, dml_data.t_col
    t_values = dml_data.t_values

    expected = _expected_n_obs_subset(data, g_col, t_col, t_values, g_val, t_pre, t_eval, "not_yet_treated")
    obj = _make_model(dml_data, g_val, t_pre, t_eval, "not_yet_treated")
    assert obj.n_obs_subset == expected

    # g=2 (intermediate group, treated between eval_t=1 and g_value=3) must not be in control
    control_g_vals = obj.data_subset.loc[obj.data_subset["C_indicator"] == 1, g_col].unique()
    assert 2.0 not in control_g_vals


@pytest.mark.ci
def test_not_yet_treated_excludes_intermediate_groups_pre_treatment():
    """Regression test: pre-treatment evaluation excludes groups treated between eval_t and g_value.

    With g_value=3 and t_value_eval=1, units with g=2 (treated at period 2, between
    eval_t=1 and g_value=3) must NOT appear in the not_yet_treated control group.
    The comparison_period is max(eval_t, g_value)=3, so only units with g > 3 (i.e.,
    never-treated) are eligible as later-treated control units.
    """
    g_val, t_pre, t_eval = 3.0, 0, 1
    g_col = dml_data.g_col

    obj_nyt = _make_model(dml_data, g_val, t_pre, t_eval, "not_yet_treated")
    obj_nt = _make_model(dml_data, g_val, t_pre, t_eval, "never_treated")

    # Both control groups end up with the same units: only never-treated (g=inf)
    assert obj_nyt.n_obs_subset == obj_nt.n_obs_subset

    # g=2 must be absent from the not_yet_treated control
    control_g_vals = obj_nyt.data_subset.loc[obj_nyt.data_subset["C_indicator"] == 1, g_col].unique()
    assert 2.0 not in control_g_vals

    # never-treated units are present
    assert any(np.isinf(v) for v in control_g_vals)


@pytest.mark.ci
def test_not_yet_treated_includes_later_treated_post_treatment():
    """Post-treatment not_yet_treated control includes groups treated after the evaluation period.

    With n_periods=5 data (g in {2,3,4,inf}), g_value=2, t_value_eval=3:
    comparison_period = max(3,2) = 3, max_g_value = 3.
    Units with g=4 (later-treated) enter the control; g=3 (treated before eval_t) does not.
    """
    g_val, t_pre, t_eval = 2.0, 0, 3
    g_col, t_col = dml_data5.g_col, dml_data5.t_col
    t_values = dml_data5.t_values
    data = dml_data5.data

    expected = _expected_n_obs_subset(data, g_col, t_col, t_values, g_val, t_pre, t_eval, "not_yet_treated")
    obj = _make_model(dml_data5, g_val, t_pre, t_eval, "not_yet_treated")
    assert obj.n_obs_subset == expected

    control_g_vals = obj.data_subset.loc[obj.data_subset["C_indicator"] == 1, g_col].unique()
    assert 4.0 in control_g_vals  # later-treated group IS in control
    assert 3.0 not in control_g_vals  # treated before eval_t, NOT in control
