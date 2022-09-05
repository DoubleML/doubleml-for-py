import numpy as np
import statsmodels.api as sm
from scipy.linalg import sqrtm
from scipy.stats import norm
import pandas as pd
import patsy


def fit_blp(obj_dml_irm, basis):
    orth_signal = obj_dml_irm.psi_b.reshape(-1, 1)
    blp_model = sm.OLS(orth_signal, basis).fit()

    return blp_model


def blp_confint(obj_dml_irm, blp_model, basis, joint=False, level=0.95, n_rep_boot=500):
    alpha = 1 - level
    g_hat = blp_model.predict(basis)

    blp_omega = blp_model.cov_HC0

    # check this again (calculation of HC0 should include scaling with sample size)
    # se_scaling = np.sqrt(self._dml_irm._dml_data.n_obs)
    se_scaling = 1
    blp_se = np.sqrt((basis.dot(blp_omega) * basis).sum(axis=1)) / se_scaling

    if joint:
        # calculate the maximum t-statistic with bootstrap
        normal_samples = np.random.normal(size=[basis.shape[1], n_rep_boot])
        bootstrap_samples = np.multiply(basis.dot(np.dot(sqrtm(blp_omega), normal_samples)).T,
                                        (blp_se * se_scaling))

        max_t_stat = np.quantile(np.max(np.abs(bootstrap_samples), axis=0), q=level)

        # Lower simultaneous CI
        g_hat_lower = g_hat - max_t_stat * blp_se
        # Upper simultaneous CI
        g_hat_upper = g_hat + max_t_stat * blp_se

    else:
        # Lower point-wise CI
        g_hat_lower = g_hat + norm.ppf(q=alpha / 2) * blp_se
        # Upper point-wise CI
        g_hat_upper = g_hat + norm.ppf(q=1 - alpha / 2) * blp_se

    ci = np.vstack((g_hat_lower, g_hat, g_hat_upper)).T
    df_ci = pd.DataFrame(ci,
                         columns=['{:.1f} %'.format(alpha / 2 * 100), 'effect',
                                  '{:.1f} %'.format((1 - alpha / 2) * 100)],
                         index=basis.index)
    return df_ci


def create_spline_basis(X, knots=5, degree=3):
    if type(knots) == int:
        X_splines = patsy.bs(X, df=knots + degree, degree=degree)
    else:
        X_splines = patsy.bs(X, knots=knots, degree=degree)
    return X_splines


def create_synthetic_data(n=200, n_w=30, support_size=5, n_x=1, constant=True):
    """
    Creates a synthetic example based on example 2 of https://github.com/microsoft/EconML/blob/master/notebooks/Double%20Machine%20Learning%20Examples.ipynb

    Parameters
    ----------
    n_samples : int
        Number of samples.
        Default is ``200``.

    n_w : int
        Dimension of covariates.
        Default is ``30``.

    support_size : int
        Number of relevant covariates.
        Default is ``5``.

    n_x : int
        Dimension of treatment variable.
        Default is ``1``.

    Returns
    -------
     data : pd.DataFrame
            A data frame.

    """
    # Outcome support
    # With the next two lines we are effectively choosing the matrix gamma in the example
    support_Y = np.random.choice(np.arange(n_w), size=support_size, replace=False)
    coefs_Y = np.random.uniform(0, 1, size=support_size)
    # Define the function to generate the noise
    epsilon_sample = lambda n: np.random.uniform(-1, 1, size=n)
    # Treatment support
    # Assuming the matrices gamma and beta have the same non-zero components
    support_T = support_Y
    coefs_T = np.random.uniform(0, 1, size=support_size)
    # Define the function to generate the noise
    eta_sample = lambda n: np.random.uniform(-1, 1, size=n)

    # Generate controls, covariates, treatments and outcomes
    W = np.random.normal(0, 1, size=(n, n_w))
    X = np.random.uniform(0, 1, size=(n, n_x))
    # Heterogeneous treatment effects
    if constant:
        TE = np.array([3 for x_i in X])
    else:
        TE = np.array([np.exp(4 + 2 * x_i) for x_i in X])
    # Define treatment
    log_odds = np.dot(W[:, support_T], coefs_T) + eta_sample(n)
    T_sigmoid = 1 / (1 + np.exp(-log_odds))
    T = np.array([np.random.binomial(1, p) for p in T_sigmoid])
    # Define the outcome
    Y = TE * T + np.dot(W[:, support_Y], coefs_Y) + epsilon_sample(n)

    # Now we build the dataset
    y_df = pd.DataFrame({'y': Y})
    x_df = pd.DataFrame({'x': X.reshape(-1)})
    t_df = pd.DataFrame({'t': T})
    w_df = pd.DataFrame(data=W, index=np.arange(W.shape[0]), columns=[f'w_{i}' for i in range(W.shape[1])])

    data = pd.concat([y_df, x_df, t_df, w_df], axis=1)

    covariates = list(w_df.columns.values) + list(x_df.columns.values)
    return data, covariates