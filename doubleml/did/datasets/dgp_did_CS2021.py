import numpy as np
import pandas as pd

from .dgp_did_SZ2020 import _generate_features, _select_features

# Based on https://doi.org/10.1016/j.jeconom.2020.12.001 (see Appendix SC)
# and https://d2cml-ai.github.io/csdid/examples/csdid_basic.html#Examples-with-simulated-data


def _f_ps_groups(w, xi, n_groups):
    # Create coefficient matrix: 4 features x n_groups
    coef_vec = np.array([-1.0, 0.5, -0.25, -0.2])

    # use i_group/n_groups as coeffect for columns
    coef_matrix = np.array([coef_vec * (1.0 - (i_group / n_groups)) for i_group in range(n_groups)]).T

    res = xi * (w @ coef_matrix)
    return res


def _f_reg_time(w, n_time_periods):
    coef_vec = np.array([27.4, 13.7, 13.7, 13.7])

    # use time period as coeffect for columns
    coef_matrix = np.array([coef_vec * (i_time / n_time_periods) for i_time in range(1, n_time_periods + 1)]).T

    res = 210 + w @ coef_matrix
    return res


def make_did_CS2021(n_obs=1000, dgp_type=1, include_never_treated=True, time_type="datetime", **kwargs):
    """
    Generate synthetic panel data for difference-in-differences analysis based on Callaway and Sant'Anna (2021).

    This function creates panel data with heterogeneous treatment effects across time periods and groups.
    The data includes pre-treatment periods, multiple treatment groups that receive treatment at different times,
    and optionally a never-treated group that serves as a control. The true average treatment effect on the
    treated (ATT) has a heterogeneous structure dependent on covariates and exposure time.

    The data generating process offers six variations (``dgp_type`` 1-6) that differ in how the regression features
    and propensity score features are derived:

    - DGP 1: Outcome and propensity score are linear (in Z)
    - DGP 2: Outcome is linear, propensity score is nonlinear
    - DGP 3: Outcome is nonlinear, propensity score is linear
    - DGP 4: Outcome and propensity score are nonlinear
    - DGP 5: Outcome is linear, propensity score is constant (experimental setting)
    - DGP 6: Outcome is nonlinear, propensity score is constant (experimental setting)

    Let :math:`X= (X_1, X_2, X_3, X_4)^T \\sim \\mathcal{N}(0, \\Sigma)`, where :math:`\\Sigma` is a matrix with entries
    :math:`\\Sigma_{kj} = c^{|j-k|}`. The default value is :math:`c = 0`, corresponding to the identity matrix.

    Further, define :math:`Z_j = (\\tilde{Z_j} - \\mathbb{E}[\\tilde{Z}_j]) / \\sqrt{\\text{Var}(\\tilde{Z}_j)}`,
    where :math:`\\tilde{Z}_1 = \\exp(0.5 \\cdot X_1)`, :math:`\\tilde{Z}_2 = 10 + X_2/(1 + \\exp(X_1))`,
    :math:`\\tilde{Z}_3 = (0.6 + X_1 \\cdot X_3 / 25)^3` and :math:`\\tilde{Z}_4 = (20 + X_2 + X_4)^2`.

    For a feature vector :math:`W=(W_1, W_2, W_3, W_4)^T` (either X or Z based on ``dgp_type``), the core functions are:

    1. Time-varying outcome regression function for each time period :math:`t`:

       .. math::

           f_{reg,t}(W) = 210 + \\frac{t}{T} \\cdot (27.4 \\cdot W_1 + 13.7 \\cdot W_2 + 13.7 \\cdot W_3 + 13.7 \\cdot W_4)

    2. Group-specific propensity function for each treatment group :math:`g`:

       .. math::

           f_{ps,g}(W) = \\xi \\cdot \\left(1-\\frac{g}{G}\\right) \\cdot
           (-W_1 + 0.5 \\cdot W_2 - 0.25 \\cdot W_3 - 0.2\\cdot W_4)

    where :math:`T` is the number of time periods, :math:`G` is the number of treatment groups, and :math:`\\xi` is a
    scale parameter (default: 0.9).

    The panel data model is defined with the following components:

    1. Time effects: :math:`\\delta_t = t` for time period :math:`t`

    2. Individual effects: :math:`\\eta_i \\sim \\mathcal{N}(g_i, 1)` where :math:`g_i` is unit :math:`i`'s treatment group

    3. Treatment effects: For a unit in treatment group :math:`g`, the effect in period :math:`t` is:

       .. math::

           \\theta_{i,t,g} = \\max(t - t_g + 1, 0) + 0.1 \\cdot X_{i,1} \\cdot \\max(t - t_g + 1, 0)

       where :math:`t_g` is the first treatment period for group :math:`g`, :math:`X_{i,1}` is the first covariate for unit
       :math:`i`, and :math:`\\max(t - t_g + 1, 0)` represents the exposure time (0 for pre-treatment periods).

    4. Potential outcomes for unit :math:`i` in period :math:`t`:

       .. math::

           Y_{i,t}(0) &= f_{reg,t}(W_{reg}) + \\delta_t + \\eta_i + \\varepsilon_{i,0,t}

           Y_{i,t}(1) &= Y_{i,t}(0) + \\theta_{i,t,g} + (\\varepsilon_{i,1,t} - \\varepsilon_{i,0,t})

       where :math:`\\varepsilon_{i,0,t}, \\varepsilon_{i,1,t} \\sim \\mathcal{N}(0, 1)`.

    5. Observed outcomes:

       .. math::

           Y_{i,t} = Y_{i,t}(1) \\cdot 1\\{t \\geq t_g\\} + Y_{i,t}(0) \\cdot 1\\{t < t_g\\}

    6. Treatment assignment:

       For non-experimental settings (DGP 1-4), the probability of being in treatment group :math:`g` is:

       .. math::

           P(G_i = g) = \\frac{\\exp(f_{ps,g}(W_{ps}))}{\\sum_{g'} \\exp(f_{ps,g'}(W_{ps}))}

       For experimental settings (DGP 5-6), each treatment group (including never-treated) has equal probability:

       .. math::

           P(G_i = g) = \\frac{1}{G} \\text{ for all } g

    The variables :math:`W_{reg}` and :math:`W_{ps}` are selected based on the DGP type:

    .. math::

        DGP1:\\quad W_{reg} &= Z \\quad W_{ps} = Z

        DGP2:\\quad W_{reg} &= Z \\quad W_{ps} = X

        DGP3:\\quad W_{reg} &= X \\quad W_{ps} = Z

        DGP4:\\quad W_{reg} &= X \\quad W_{ps} = X

        DGP5:\\quad W_{reg} &= Z \\quad W_{ps} = 0

        DGP6:\\quad W_{reg} &= X \\quad W_{ps} = 0

    where settings 5-6 correspond to experimental designs with equal probability across treatment groups.


    Parameters
    ----------
    n_obs : int, default=1000
        The number of observations to simulate.

    dgp_type : int, default=1
        The data generating process to be used (1-6).

    include_never_treated : bool, default=True
        Whether to include units that are never treated.

    time_type : str, default="datetime"
        Type of time variable. Either "datetime" or "float".

    **kwargs
        Additional keyword arguments. Accepts the following parameters:

        `c` (float, default=0.0):
            Parameter for correlation structure in X.

        `dim_x` (int, default=4):
            Dimension of feature vectors.

        `xi` (float, default=0.9):
            Scale parameter for the propensity score function.

        `n_periods` (int, default=5):
            Number of time periods.

        `anticipation_periods` (int, default=0):
            Number of periods before treatment where anticipation effects occur.

        `n_pre_treat_periods` (int, default=2):
            Number of pre-treatment periods.

        `start_date` (str, default="2025-01"):
            Start date for datetime time variables.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the simulated panel data.

    References
    ----------
    Callaway, B. and Sant’Anna, P. H. (2021),
    Difference-in-Differences with multiple time periods. Journal of Econometrics, 225(2), 200-230.
    doi:`10.1016/j.jeconom.2020.12.001 <https://doi.org/10.1016/j.jeconom.2020.12.001>`_.
    """

    c = kwargs.get("c", 0.0)
    dim_x = kwargs.get("dim_x", 4)
    xi = kwargs.get("xi", 0.9)
    n_periods = kwargs.get("n_periods", 5)
    anticipation_periods = kwargs.get("anticipation_periods", 0)
    n_pre_treat_periods = kwargs.get("n_pre_treat_periods", 2)
    start_date = kwargs.get("start_date", "2025-01")

    if anticipation_periods > 0:
        n_periods += anticipation_periods  # increase number of periods

    expected_time_types = ("datetime", "float")
    if time_type not in expected_time_types:
        raise ValueError(f"time_type must be one of {expected_time_types}. Got {time_type}.")

    x, z = _generate_features(n_obs, c, dim_x=dim_x)
    features_ps, features_reg = _select_features(dgp_type, x, z)

    # generate possible time periods
    if time_type == "datetime":
        time_periods = np.array([np.datetime64(start_date) + np.timedelta64(i, "M") for i in range(n_periods)])
        never_treated_value = np.datetime64("NaT")
    else:
        assert time_type == "float"
        time_periods = np.arange(n_periods)
        never_treated_value = np.inf
    n_time_periods = len(time_periods)

    # set treatment values for time periods greater than n_pre_treat_periods
    treatment_values = time_periods[time_periods >= time_periods[n_pre_treat_periods]]
    max_exposure = len(treatment_values)  # exclude never treated
    if include_never_treated:
        treatment_values = np.append(treatment_values, never_treated_value)
    n_treatment_groups = len(treatment_values)

    # treatment assignment and propensities (shape (n_obs,))
    is_experimental = (dgp_type == 5) or (dgp_type == 6)
    if is_experimental:
        # Set D to be experimental
        p = np.ones(n_treatment_groups) / n_treatment_groups
        d_index = np.random.choice(n_treatment_groups, size=n_obs, p=p)
    else:
        unnormalized_p = np.exp(_f_ps_groups(features_ps, xi, n_groups=n_treatment_groups))
        p = unnormalized_p / unnormalized_p.sum(1, keepdims=True)
        d_index = np.array([np.random.choice(n_treatment_groups, p=p_row) for p_row in p])

    # fixed effects (shape (n_obs, n_time_periods))
    time_effects = np.arange(n_time_periods)
    delta_t = np.tile(time_effects, (n_obs, 1))
    indiviual_effects = np.random.normal(loc=d_index, scale=1, size=(n_obs,))
    eta_i = np.tile(indiviual_effects, (n_time_periods, 1)).T

    # error terms (shape (n_obs, n_time_periods))
    epsilon_0 = np.random.normal(loc=0, scale=1, size=(n_obs, n_time_periods))
    epsilon_1 = np.random.normal(loc=0, scale=1, size=(n_obs, n_time_periods))

    # regression function (shape (n_obs, n_time_periods))
    f_reg = _f_reg_time(features_reg, n_time_periods)

    # treatment effecs (shape (n_obs, n_time_periods))
    exposure_pre_period = np.zeros((n_obs, n_pre_treat_periods))
    exposure_post_first_treatment = np.clip(np.arange(max_exposure) - d_index.reshape(-1, 1) + 1, a_min=0, a_max=None)
    exposure_time = np.column_stack((exposure_pre_period, exposure_post_first_treatment))
    delta_e = exposure_time

    # add heterogeneity in treatment effects
    heterogeneity_x = 0.1 * x[:, 0]
    heterogeneity = heterogeneity_x.reshape(-1, 1) * exposure_time
    delta_e += heterogeneity

    # potential outcomes (shape (n_obs, n_time_periods))
    y0 = f_reg + delta_t + eta_i + epsilon_0
    y1 = y0 + delta_e + (epsilon_1 - epsilon_0)

    # observed outcomes (shape (n_obs, n_time_periods))
    is_exposed = exposure_time > 0
    y = y1 * is_exposed + y0 * ~is_exposed

    # map treatment index to values
    d = np.array([treatment_values[i] for i in d_index])
    d_matrix = np.tile(d, (n_time_periods, 1)).T

    # create matrices to flatten the data
    id_matrix = np.tile(np.arange(n_obs), (n_time_periods, 1)).T
    time_matrix = np.tile(time_periods, (n_obs, 1))

    df = pd.DataFrame(
        {
            "id": id_matrix.flatten(),
            "y": y.flatten(),
            "y0": y0.flatten(),
            "y1": y1.flatten(),
            "d": d_matrix.flatten(),
            "t": time_matrix.flatten(),
            **{f"Z{i + 1}": z[:, i].repeat(n_time_periods) for i in range(dim_x)},
        }
    )
    if anticipation_periods > 0:
        # filter time periods
        df = df[df["t"] >= time_periods[anticipation_periods]]
        # filter treatment after anticipation periods
        df = df[(df["d"] <= time_periods[-(anticipation_periods + 1)]) | pd.isna(df["d"])]

        # update time periods by subtracting time delta
        if time_type == "datetime":
            df = df[(df["d"] <= time_periods[-(anticipation_periods + 1)]) | pd.isna(df["d"])]
            df["t"] = df["t"].apply(lambda x: x - pd.DateOffset(months=anticipation_periods))
        else:
            assert time_type == "float"
            df = df[(df["d"] <= time_periods[-(anticipation_periods + 1)]) | np.isinf(df["d"])]
            df["t"] = df["t"] - anticipation_periods

    return df
