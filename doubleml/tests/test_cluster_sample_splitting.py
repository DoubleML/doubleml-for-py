"""Check inference and resampling after externally setting cluster sample splits."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from doubleml import DoubleMLData, DoubleMLPLR


@pytest.fixture(scope="module", params=[1, 2], ids=["one-way", "two-way"])
def cluster_data(request: pytest.FixtureRequest) -> DoubleMLData:
    """Generate balanced clusters with a small partially linear regression problem."""
    rng = np.random.default_rng(3141)
    n_clusters = 12
    cluster_vars = np.column_stack((np.repeat(np.arange(n_clusters), n_clusters), np.tile(np.arange(n_clusters), n_clusters)))[
        :, : request.param
    ]
    n_obs = len(cluster_vars)
    x = rng.normal(size=(n_obs, 3))
    cluster_effect = rng.normal(size=(n_clusters, request.param))
    shared_noise = cluster_effect[cluster_vars, np.arange(request.param)].sum(axis=1)
    d = x[:, 0] + 0.5 * shared_noise + rng.normal(size=n_obs)
    y = 0.5 * d + x[:, 1] + shared_noise + rng.normal(size=n_obs)
    return DoubleMLData.from_arrays(x, y, d, cluster_vars=cluster_vars)


@pytest.mark.ci
@pytest.mark.parametrize("constructor_folds, supplied_folds", [(2, 3), (3, 2)])
@pytest.mark.parametrize("n_rep", [1, 2])
def test_external_cluster_splits_inference(
    cluster_data: DoubleMLData, constructor_folds: int, supplied_folds: int, n_rep: int
) -> None:
    """Identical supplied cluster splits give identical inference regardless of initial fold counts."""
    np.random.seed(3141)
    reference = DoubleMLPLR(cluster_data, LinearRegression(), LinearRegression(), n_folds=supplied_folds, n_rep=n_rep).fit()
    external = DoubleMLPLR(
        cluster_data,
        LinearRegression(),
        LinearRegression(),
        n_folds=constructor_folds,
        draw_sample_splitting=False,
    )
    all_smpls = reference.smpls[0] if n_rep == 1 else reference.smpls
    external.set_sample_splitting(all_smpls, reference.smpls_cluster).fit()

    np.testing.assert_allclose(external.all_coef, reference.all_coef, rtol=1e-12)
    if cluster_data.n_cluster_vars == 1:
        _, cluster_ids = np.unique(cluster_data.cluster_vars[:, 0], return_inverse=True)
        for i_rep in range(n_rep):
            cluster_scores = np.bincount(cluster_ids, weights=external.psi[:, i_rep, 0])
            se_cr0 = np.sqrt(np.sum(cluster_scores**2)) / abs(external.psi_deriv[:, i_rep, 0].sum())
            np.testing.assert_allclose(external.all_se[0, i_rep], se_cr0, rtol=1e-12)
    np.testing.assert_allclose(external.all_se, reference.all_se, rtol=1e-12)
    np.testing.assert_allclose(external.coef, reference.coef, rtol=1e-12)
    np.testing.assert_allclose(external.se, reference.se, rtol=1e-12)


@pytest.mark.ci
@pytest.mark.parametrize("constructor_folds, supplied_folds", [(2, 3), (3, 2)])
def test_external_cluster_splits_redraw(cluster_data: DoubleMLData, constructor_folds: int, supplied_folds: int) -> None:
    """Redrawing uses the externally supplied cluster-fold count and repetitions."""
    np.random.seed(3141)
    reference = DoubleMLPLR(cluster_data, LinearRegression(), LinearRegression(), n_folds=supplied_folds, n_rep=2)
    external = DoubleMLPLR(
        cluster_data,
        LinearRegression(),
        LinearRegression(),
        n_folds=constructor_folds,
        draw_sample_splitting=False,
    )
    external.set_sample_splitting(reference.smpls, reference.smpls_cluster)
    external.draw_sample_splitting()

    expected_folds = supplied_folds**cluster_data.n_cluster_vars
    assert external.n_folds == expected_folds
    assert external.n_rep == 2
    assert len(external.smpls) == external.n_rep
    assert len(external.smpls_cluster) == external.n_rep
    for smpls, smpls_cluster in zip(external.smpls, external.smpls_cluster):
        assert len(smpls) == expected_folds
        assert len(smpls_cluster) == expected_folds


@pytest.mark.ci
@pytest.mark.parametrize("cluster_data", [2], indirect=True)
def test_external_cluster_splits_rectangular_grid(cluster_data: DoubleMLData) -> None:
    """Reject a rectangular grid that cannot use one common fold count per cluster variable."""
    cluster_vars = cluster_data.cluster_vars
    clusters = [np.unique(cluster_vars[:, i_var]) for i_var in range(2)]
    smpls = []
    smpls_cluster = []
    for first_test in np.array_split(clusters[0], 2):
        for second_test in np.array_split(clusters[1], 3):
            train_clusters = [np.setdiff1d(clusters[0], first_test), np.setdiff1d(clusters[1], second_test)]
            test_clusters = [first_test, second_test]
            train_mask = np.isin(cluster_vars[:, 0], train_clusters[0]) & np.isin(cluster_vars[:, 1], train_clusters[1])
            test_mask = np.isin(cluster_vars[:, 0], first_test) & np.isin(cluster_vars[:, 1], second_test)
            smpls.append((np.flatnonzero(train_mask), np.flatnonzero(test_mask)))
            smpls_cluster.append((train_clusters, test_clusters))

    model = DoubleMLPLR(cluster_data, LinearRegression(), LinearRegression(), draw_sample_splitting=False)
    msg = r"Invalid cluster sample splitting\. n_folds must be a power of n_folds_per_cluster"
    with pytest.raises(ValueError, match=msg):
        model.set_sample_splitting(smpls, [smpls_cluster])
