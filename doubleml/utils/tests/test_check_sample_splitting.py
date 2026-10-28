"""Test the validation of externally supplied cluster sample splits."""

import numpy as np
import pytest

from doubleml import DoubleMLData
from doubleml.utils._checks import _check_sample_splitting
from doubleml.utils.resampling import DoubleMLClusterResampling

N_CLUSTERS = 6


def _cluster_data(n_cluster_vars: int) -> DoubleMLData:
    """Generate a small data set with balanced one- or two-way clusters."""
    rng = np.random.default_rng(42)
    grid = np.column_stack((np.repeat(np.arange(N_CLUSTERS), N_CLUSTERS), np.tile(np.arange(N_CLUSTERS), N_CLUSTERS)))
    n_obs = len(grid)
    return DoubleMLData.from_arrays(
        rng.normal(size=(n_obs, 2)), rng.normal(size=n_obs), rng.normal(size=n_obs), cluster_vars=grid[:, :n_cluster_vars]
    )


def _draw_cluster_splits(dml_data: DoubleMLData, n_folds: int = 3, n_rep: int = 2) -> tuple[list, list]:
    """Draw repeated cluster sample splits."""
    return DoubleMLClusterResampling(
        n_folds=n_folds,
        n_rep=n_rep,
        n_obs=dml_data.n_obs,
        n_cluster_vars=dml_data.n_cluster_vars,
        cluster_vars=dml_data.cluster_vars,
    ).split_samples()


@pytest.mark.ci
@pytest.mark.parametrize("n_cluster_vars", [1, 2])
def test_check_sample_splitting_n_folds_per_cluster(n_cluster_vars: int) -> None:
    """The number of folds per cluster variable is derived from the supplied splits."""
    dml_data = _cluster_data(n_cluster_vars)
    _, _, _, n_folds, n_folds_per_cluster = _check_sample_splitting(*_draw_cluster_splits(dml_data), dml_data, True)
    assert n_folds == 3**n_cluster_vars
    assert n_folds_per_cluster == 3


@pytest.mark.ci
def test_check_sample_splitting_rectangular_grid() -> None:
    """Reject a 2 x 3 grid that cannot use one common fold count per cluster variable."""
    dml_data = _cluster_data(2)
    cluster_vars = dml_data.cluster_vars
    clusters = np.arange(N_CLUSTERS)
    smpls, smpls_cluster = [], []
    for first_test in np.array_split(clusters, 2):
        for second_test in np.array_split(clusters, 3):
            train_clusters = [np.setdiff1d(clusters, first_test), np.setdiff1d(clusters, second_test)]
            train_mask = np.isin(cluster_vars[:, 0], train_clusters[0]) & np.isin(cluster_vars[:, 1], train_clusters[1])
            test_mask = np.isin(cluster_vars[:, 0], first_test) & np.isin(cluster_vars[:, 1], second_test)
            smpls.append((np.flatnonzero(train_mask), np.flatnonzero(test_mask)))
            smpls_cluster.append((train_clusters, [first_test, second_test]))

    msg = r"Invalid cluster sample splitting\. The number of folds must equal n_folds_per_cluster \*\* n_cluster_vars"
    with pytest.raises(ValueError, match=msg):
        _check_sample_splitting(smpls, [smpls_cluster], dml_data, True)


@pytest.mark.ci
@pytest.mark.parametrize("i_rep, i_fold", [(0, 1), (1, 0)])
def test_check_sample_splitting_invalid_cluster_partition(i_rep: int, i_fold: int) -> None:
    """Cluster partitions are validated for every repetition and fold, not only the first one."""
    dml_data = _cluster_data(1)
    all_smpls, all_smpls_cluster = _draw_cluster_splits(dml_data)
    train_clusters, test_clusters = all_smpls_cluster[i_rep][i_fold]
    all_smpls_cluster[i_rep][i_fold] = (train_clusters, [test_clusters[0][1:]])

    msg = r"Invalid cluster partition provided\. At least one inner list does not form a partition\."
    with pytest.raises(ValueError, match=msg):
        _check_sample_splitting(all_smpls, all_smpls_cluster, dml_data, True)
