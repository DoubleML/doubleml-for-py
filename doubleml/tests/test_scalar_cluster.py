"""Test cluster-based sample splitting for scalar PLR models."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from doubleml import DoubleMLData
from doubleml.plm.datasets import make_plr_CCDDHNR2018
from doubleml.plm.plr_scalar import PLR

from ._utils import _clone


@pytest.fixture(
    scope="module",
    params=[
        RandomForestRegressor(max_depth=2, n_estimators=10, random_state=42),
        LinearRegression(),
    ],
)
def learner(request):
    return request.param


@pytest.mark.ci
def test_scalar_plr_cluster_set_sample_splitting():
    """Check set_sample_splitting consistency for scalar PLR cluster data."""
    np.random.seed(3141)
    n_i = 5
    n_j = 6
    n_obs = n_i * n_j

    df = make_plr_CCDDHNR2018(n_obs=n_obs, return_type="DataFrame")
    x_cols = [col for col in df.columns if col.startswith("X")]

    df["cluster_i"] = np.repeat(np.arange(n_i), n_j)
    df["cluster_j"] = np.tile(np.arange(n_j), n_i)

    dml_data = DoubleMLData(df, y_col="y", d_cols="d", x_cols=x_cols, cluster_cols=["cluster_i", "cluster_j"])

    ml_l = LinearRegression()
    ml_m = LinearRegression()

    dml_obj = PLR(dml_data)
    dml_obj.set_learners(ml_l=ml_l, ml_m=ml_m)
    dml_obj.draw_sample_splitting(n_folds=2, n_rep=2)
    dml_obj.fit()

    dml_obj_ext = PLR(dml_data)
    dml_obj_ext.set_learners(ml_l=LinearRegression(), ml_m=LinearRegression())
    dml_obj_ext.set_sample_splitting(all_smpls=dml_obj.smpls, all_smpls_cluster=dml_obj.smpls_cluster)
    dml_obj_ext.fit()

    assert np.isclose(dml_obj.coef[0], dml_obj_ext.coef[0], rtol=1e-9, atol=1e-4)
    assert np.isclose(dml_obj.se[0], dml_obj_ext.se[0], rtol=1e-9, atol=1e-4)


@pytest.fixture(scope="module")
def dml_plr_scalar_cluster_with_index(generate_data1, learner):
    """Fit scalar PLR with and without clustering for comparison."""
    # in the one-way cluster case with exactly one observation per cluster, we get the same result w & w/o clustering
    n_folds = 2

    data = generate_data1
    x_cols = data.columns[data.columns.str.startswith("X")].tolist()

    ml_l = _clone(learner)
    ml_m = _clone(learner)

    obj_dml_data = DoubleMLData(data, "y", ["d"], x_cols)
    np.random.seed(3141)
    dml_plr_obj = PLR(obj_dml_data)
    dml_plr_obj.set_learners(ml_l=ml_l, ml_m=ml_m)
    dml_plr_obj.draw_sample_splitting(n_folds=n_folds)
    dml_plr_obj.fit()

    df = data.reset_index()
    dml_cluster_data = DoubleMLData(df, y_col="y", d_cols="d", x_cols=x_cols, cluster_cols="index")
    np.random.seed(3141)
    dml_plr_cluster_obj = PLR(dml_cluster_data)
    dml_plr_cluster_obj.set_learners(ml_l=_clone(learner), ml_m=_clone(learner))
    dml_plr_cluster_obj.draw_sample_splitting(n_folds=n_folds)
    dml_plr_cluster_obj.fit()

    dml_plr_cluster_ext_smpls = PLR(dml_cluster_data)
    dml_plr_cluster_ext_smpls.set_learners(ml_l=_clone(learner), ml_m=_clone(learner))
    dml_plr_cluster_ext_smpls.set_sample_splitting(
        all_smpls=dml_plr_cluster_obj.smpls,
        all_smpls_cluster=dml_plr_cluster_obj.smpls_cluster,
    )
    np.random.seed(3141)
    dml_plr_cluster_ext_smpls.fit()

    res_dict = {
        "coef": dml_plr_obj.coef,
        "coef_manual": dml_plr_cluster_obj.coef,
        "se": dml_plr_obj.se,
        "se_manual": dml_plr_cluster_obj.se,
        "coef_ext_smpls": dml_plr_cluster_ext_smpls.coef,
        "se_ext_smpls": dml_plr_cluster_ext_smpls.se,
    }

    return res_dict


@pytest.mark.ci
def test_dml_plr_scalar_cluster_with_index_coef(dml_plr_scalar_cluster_with_index):
    """Validate scalar PLR cluster coefficients match across configurations."""
    assert np.isclose(
        dml_plr_scalar_cluster_with_index["coef"][0],
        dml_plr_scalar_cluster_with_index["coef_manual"][0],
        rtol=1e-9,
        atol=1e-4,
    )
    assert np.isclose(
        dml_plr_scalar_cluster_with_index["coef"][0],
        dml_plr_scalar_cluster_with_index["coef_ext_smpls"][0],
        rtol=1e-9,
        atol=1e-4,
    )


@pytest.mark.ci
def test_dml_plr_scalar_cluster_with_index_se(dml_plr_scalar_cluster_with_index):
    """Validate scalar PLR cluster standard errors match across configurations."""
    assert np.isclose(
        dml_plr_scalar_cluster_with_index["se"][0],
        dml_plr_scalar_cluster_with_index["se_manual"][0],
        rtol=1e-9,
        atol=1e-4,
    )
    assert np.isclose(
        dml_plr_scalar_cluster_with_index["se"][0],
        dml_plr_scalar_cluster_with_index["se_ext_smpls"][0],
        rtol=1e-9,
        atol=1e-4,
    )
