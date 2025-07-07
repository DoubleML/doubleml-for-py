import numpy as np

from doubleml.did.datasets.dgp_did_CS2021 import make_did_CS2021

# Based on https://doi.org/10.1016/j.jeconom.2020.12.001 (see Appendix SC)
# and https://d2cml-ai.github.io/csdid/examples/csdid_basic.html#Examples-with-simulated-data
# Cross-sectional version of the data generating process (DGP) for Callaway and Sant'Anna (2021)


def make_did_cs_CS2021(n_obs=1000, dgp_type=1, include_never_treated=True, lambda_t=0.5, time_type="datetime", **kwargs):
    """
    Generate synthetic repeated cross-sectional data for difference-in-differences analysis based on
    Callaway and Sant'Anna (2021).

    This function creates repeated cross-sectional data with heterogeneous treatment effects across time periods and groups.
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

    7. Steps 1-6 generate panel data. To obtain repeated cross-sectional data, the number of generated individuals is increased
    to `n_obs/lambda_t`, where `lambda_t` denotes the probability to observe a unit at each time period (time constant).
    for each


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

    lambda_t : float, default=0.5
        Probability of observing a unit at each time period. Note that internally `n_obs/lambda_t` individuals are
        generated of which only a fraction `lambda_t` is observed at each time period (see Step 7 in the DGP description).

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

    n_obs_panel = int(np.ceil(n_obs / lambda_t))
    df_panel = make_did_CS2021(
        n_obs=n_obs_panel,
        dgp_type=dgp_type,
        include_never_treated=include_never_treated,
        time_type=time_type,
        **kwargs,
    )

    # for each time period, randomly select units to observe
    observed_units = np.random.binomial(1, lambda_t, size=(len(df_panel.index)))
    df_repeated_cs = df_panel[observed_units == 1].copy()

    return df_repeated_cs
