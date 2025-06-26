import numpy as np
import pandas as pd


def make_static_panel_CP2025(num_n=250, num_t=10, dim_x=30, theta=0.5, dgp_type='dgp1', x_var=5, a_var=0.95):
    """
    Generates static panel data from the simulation dgp in Clarke and Polselli (2025).

    Parameters
    ----------
    num_n :
        The number of unit in the panel.
    num_t :
        The number of time periods in the panel.
    num_x :
        The number of of covariates.
    theta :
        The value of the causal parameter.
    dgp_type :
        The type of DGP design to be used. Default is ``'dgp1'``, other options are ``'dgp2'`` and ``'dgp3'``.
    x_var : 
        The variance of the covariates.
    a_var :
        The variance of the individual fixed effect on outcome

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the simulated static panel data.
    """

    # parameters
    a = 0.25
    b = 0.5

    # id and time vectors
    id = np.repeat(np.arange(1, num_n+1), num_t)
    time = np.tile(np.arange(1, num_t+1), num_n)

    # individual fixed effects
    a_i = np.repeat(np.random.normal(0, np.sqrt(a_var), num_n), num_t)
    c_i = np.repeat(np.random.standard_normal(num_n), num_t)

    # covariates and errors
    x_mean = 0
    x_it = np.random.normal(loc=x_mean, scale=np.sqrt(x_var), size=(num_n*num_t, dim_x))
    u_it = np.random.standard_normal(num_n*num_t)
    v_it = np.random.standard_normal(num_n*num_t)

    # functional forms in nuisance functions
    if dgp_type == 'dgp1':
        l_0 = a * x_it[:,0] + x_it[:,2]
        m_0 = a * x_it[:,0] + x_it[:,2]
    elif dgp_type == 'dgp2':
        l_0 = np.divide(np.exp(x_it[:,0]), 1 + np.exp(x_it[:,0])) + a * np.cos(x_it[:,2])
        m_0 = np.cos(x_it[:,0]) + a * np.divide(np.exp(x_it[:,2]), 1 + np.exp(x_it[:,2]))
    elif dgp_type == 'dgp3':
        l_0 = b * (x_it[:,0] * x_it[:,2]) + a * (x_it[:,2] * np.where(x_it[:,2] > 0, 1, 0))
        m_0 = a * (x_it[:,0] * np.where(x_it[:,0] > 0, 1, 0)) + b * (x_it[:,0] * x_it[:,2])
    else:
        raise ValueError('Invalid dgp')
    
    # treatment
    d_it = m_0 + c_i + v_it

    def alpha_i(x_it, d_it, a_i, num_n, num_t):
        d_i = np.array_split(d_it, num_n)
        d_i_term = np.repeat(np.mean(d_i, axis=1), num_t) - np.mean(d_it)

        x_i = np.array_split(np.sum(x_it[:, [0, 2]], axis=1), num_n)
        x_i_mean = np.mean(x_i, axis=1)
        x_i_term = np.repeat(x_i_mean, num_t)

        alpha_term = 0.25 * d_i_term + 0.25 * x_i_term + a_i
        return alpha_term
    
    # outcome
    y_it = d_it * theta + l_0 + alpha_i(x_it, d_it, a_i, num_n, num_t) + u_it

    x_cols = [f'x{i + 1}' for i in np.arange(dim_x)]

    data = pd.DataFrame(np.column_stack((id, time, d_it, y_it, x_it)),
                        columns=['id', 'time', 'd', 'y'] + x_cols).astype({'id': 'int64', 'time': 'int64'})
    
    return data