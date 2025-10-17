import numpy as np
import pandas as pd
from scipy.linalg import toeplitz


def make_plr_cluster_data(
    n_clusters1=50,
    n_clusters2=None,
    obs_per_cluster=10,
    dim_x=5,
    alpha=0.5,
    cluster_correlation=0.5,
    error_correlation=0.3,
    linear=False,
    cluster_size_variation=0.3,
    **kwargs,
):
    """
    Generates clustered data from a partially linear regression model with potential outcomes.

    For one-way clustering:
    .. math::
        d_{ic} = m_0(x_{ic}) + \\xi_c^d + v_{ic}
        y_{ic}(d) = \\alpha d + g_0(x_{ic}) + \\xi_c^y + \\epsilon_{ic}

    For two-way clustering:
    .. math::
        d_{ict} = m_0(x_{ict}) + \\xi_c^d + \\zeta_t^d + v_{ict}
        y_{ict}(d) = \\alpha d + g_0(x_{ict}) + \\xi_c^y + \\zeta_t^y + \\epsilon_{ict}

    where cluster effects and errors are correlated within clusters.

    Parameters
    ----------
    n_clusters1 : int
        Number of clusters in first dimension (e.g., firms).
        Default is 50.
    n_clusters2 : int or None
        Number of clusters in second dimension (e.g., time periods).
        If None, only one-way clustering is used.
        Default is None.
    obs_per_cluster : int
        Average number of observations per first cluster dimension.
        Default is 10.
    dim_x : int
        Number of covariates.
        Default is 5.
    alpha : float
        Treatment effect parameter.
        Default is 0.5.
    cluster_correlation : float
        Correlation of variables within clusters.
        Default is 0.5.
    error_correlation : float
        Correlation of errors within clusters.
        Default is 0.3.
    linear : bool
        If True, uses linear nuisance functions. If False, uses nonlinear functions
        similar to CCDDHNR2018.
        Default is False.
    cluster_size_variation : float
        Controls variation in cluster sizes. Higher values create more variation.
        Set to 0 for constant cluster sizes.
        Default is 0.3.
    **kwargs
        Additional parameters for nuisance functions and error variances.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'data': DoubleMLData object with observed data
        - 'oracle_values': dict containing potential outcomes and nuisance functions
    """

    # Set default parameters for nuisance functions
    a_0 = kwargs.get("a_0", 1.0)
    a_1 = kwargs.get("a_1", 0.25)
    s_1 = kwargs.get("s_1", 1.0)

    b_0 = kwargs.get("b_0", 1.0)
    b_1 = kwargs.get("b_1", 0.25)
    s_2 = kwargs.get("s_2", 1.0)

    x_cov = kwargs.get("x_cov", 0.1)  # Covariance parameter for X

    # Cluster effect parameters
    sigma_xi_d = kwargs.get("sigma_xi_d", 0.3)  # Cluster effect variance for treatment
    sigma_xi_y = kwargs.get("sigma_xi_y", 0.5)  # Cluster effect variance for outcome
    sigma_zeta_d = kwargs.get("sigma_zeta_d", 0.2)  # Second cluster effect variance for treatment
    sigma_zeta_y = kwargs.get("sigma_zeta_y", 0.3)  # Second cluster effect variance for outcome

    # Generate variable cluster sizes
    def generate_cluster_sizes(n_clusters, mean_size, variation):
        if variation == 0:
            return np.full(n_clusters, int(mean_size))

        # Use gamma distribution to generate positive cluster sizes
        # Shape parameter controls variation
        shape = 1 / (variation**2)
        scale = mean_size / shape

        sizes = np.random.gamma(shape, scale, n_clusters)
        sizes = np.maximum(1, np.round(sizes).astype(int))  # At least 1 obs per cluster

        return sizes

    # Generate observations for first clustering dimension
    cluster_sizes = generate_cluster_sizes(n_clusters1, obs_per_cluster, cluster_size_variation)
    n_obs = np.sum(cluster_sizes)
    cluster1_ids = np.concatenate([np.full(size, i) for i, size in enumerate(cluster_sizes)])

    # Generate cluster effects for first dimension
    xi_d = np.random.normal(0, sigma_xi_d, n_clusters1)
    xi_y = np.random.normal(0, sigma_xi_y, n_clusters1)

    if n_clusters2 is None:
        cluster2_ids = None
        cluster_effects_d = xi_d[cluster1_ids]
        cluster_effects_y = xi_y[cluster1_ids]
    else:
        # Two-way clustering: randomly assign observations to second dimension clusters
        cluster2_ids = np.random.choice(n_clusters2, size=n_obs)

        # Generate cluster effects for second dimension
        zeta_d = np.random.normal(0, sigma_zeta_d, n_clusters2)
        zeta_y = np.random.normal(0, sigma_zeta_y, n_clusters2)

        cluster_effects_d = xi_d[cluster1_ids] + zeta_d[cluster2_ids]
        cluster_effects_y = xi_y[cluster1_ids] + zeta_y[cluster2_ids]

    # Generate correlated covariates within clusters
    cov_mat = toeplitz([np.power(x_cov, k) for k in range(dim_x)])
    x = np.zeros((n_obs, dim_x))

    if n_clusters2 is None:
        # One-way clustering: generate correlated X within clusters
        for c in range(n_clusters1):
            cluster_mask = cluster1_ids == c
            n_obs_cluster = np.sum(cluster_mask)

            # Cluster-level component
            x_cluster = np.random.multivariate_normal(np.zeros(dim_x), cov_mat)

            # Individual components
            x_individual = np.random.multivariate_normal(
                np.zeros(dim_x), (1 - cluster_correlation) * cov_mat, size=n_obs_cluster
            )

            # Combine with correlation structure
            x[cluster_mask] = np.sqrt(cluster_correlation) * x_cluster + np.sqrt(1 - cluster_correlation) * x_individual
    else:
        # Two-way clustering: generate correlated X within both cluster dimensions
        for c1 in range(n_clusters1):
            for c2 in range(n_clusters2):
                cluster_mask = (cluster1_ids == c1) & (cluster2_ids == c2)
                n_obs_cluster = np.sum(cluster_mask)

                if n_obs_cluster > 0:
                    # Cluster-level components for both dimensions
                    x_cluster1 = np.random.multivariate_normal(np.zeros(dim_x), cov_mat)
                    x_cluster2 = np.random.multivariate_normal(np.zeros(dim_x), cov_mat)

                    # Individual components
                    x_individual = np.random.multivariate_normal(
                        np.zeros(dim_x), (1 - cluster_correlation) * cov_mat, size=n_obs_cluster
                    )

                    # Combine with correlation structure
                    x[cluster_mask] = (
                        np.sqrt(cluster_correlation / 2) * x_cluster1
                        + np.sqrt(cluster_correlation / 2) * x_cluster2
                        + np.sqrt(1 - cluster_correlation) * x_individual
                    )

    # Compute nuisance functions
    if linear:
        # Linear nuisance functions
        m_0 = a_0 * x[:, 0] + a_1 * x[:, 2]
        g_0 = b_0 * x[:, 0] + b_1 * x[:, 2]
    else:
        # Nonlinear nuisance functions (similar to CCDDHNR2018)
        m_0 = a_0 * x[:, 0] + a_1 * np.divide(np.exp(x[:, 2]), 1 + np.exp(x[:, 2]))

        g_0 = b_0 * np.divide(np.exp(x[:, 0]), 1 + np.exp(x[:, 0])) + b_1 * x[:, 2]

    # Generate treatment variable with clustering
    v = np.random.standard_normal(n_obs)
    d = m_0 + cluster_effects_d + s_1 * v

    # Generate errors with within-cluster correlation
    eps = np.random.standard_normal(n_obs) * s_2

    if n_clusters2 is None:
        # One-way clustering: add within-cluster error correlation
        for c in range(n_clusters1):
            cluster_mask = cluster1_ids == c
            n_obs_cluster = np.sum(cluster_mask)
            if n_obs_cluster > 1:
                cluster_error = np.random.standard_normal()
                eps[cluster_mask] += np.sqrt(error_correlation) * s_2 * cluster_error
    else:
        # Two-way clustering: add within-cluster error correlation for both dimensions
        for c1 in range(n_clusters1):
            cluster1_error = np.random.standard_normal()
            cluster1_mask = cluster1_ids == c1
            eps[cluster1_mask] += np.sqrt(error_correlation / 2) * s_2 * cluster1_error

        for c2 in range(n_clusters2):
            cluster2_error = np.random.standard_normal()
            cluster2_mask = cluster2_ids == c2
            eps[cluster2_mask] += np.sqrt(error_correlation / 2) * s_2 * cluster2_error

    # Compute potential outcomes
    y_0 = g_0 + cluster_effects_y + eps  # Potential outcome under control
    y_1 = y_0 + alpha  # Potential outcome under treatment

    # Observed outcome
    y = d * y_1 + (1 - d) * y_0

    # Individual treatment effects
    ites = y_1 - y_0  # Always equals alpha in this linear model

    # Prepare data
    x_cols = [f"X{i + 1}" for i in range(dim_x)]

    if n_clusters2 is None:
        data_df = pd.DataFrame(np.column_stack((x, y, d, cluster1_ids)), columns=x_cols + ["y", "d", "cluster1"])
    else:
        data_df = pd.DataFrame(
            np.column_stack((x, y, d, cluster1_ids, cluster2_ids)), columns=x_cols + ["y", "d", "cluster1", "cluster2"]
        )

    oracle_values = {
        "ites": ites,
        "y_0": y_0,
        "y_1": y_1,
        "ate": alpha,
        "m_0": m_0,
        "g_0": g_0,
        "cluster_effects_d": cluster_effects_d,
        "cluster_effects_y": cluster_effects_y,
        "cluster1_ids": cluster1_ids,
        "cluster2_ids": cluster2_ids,
        "linear": linear,
    }

    if n_clusters2 is not None:
        oracle_values["xi_d"] = xi_d  # First dimension cluster effects for treatment
        oracle_values["xi_y"] = xi_y  # First dimension cluster effects for outcome
        oracle_values["zeta_d"] = zeta_d  # Second dimension cluster effects for treatment
        oracle_values["zeta_y"] = zeta_y  # Second dimension cluster effects for outcome

    return {"data": data_df, "oracle_values": oracle_values}
