import numpy as np
import statsmodels.api as sm
from scipy.linalg import sqrtm
from scipy.stats import norm
import pandas as pd


def fit_blp(orth_signal, basis, cov_type, **kwargs):
    blp_model = sm.OLS(orth_signal, basis).fit(cov_type=cov_type, **kwargs)

    return blp_model


def blp_confint(blp_model, basis, joint=False, level=0.95, n_rep_boot=500):
    alpha = 1 - level
    g_hat = blp_model.predict(basis)

    blp_omega = blp_model.cov_params().to_numpy()

    blp_se = np.sqrt((basis.dot(blp_omega) * basis).sum(axis=1))

    if joint:
        # calculate the maximum t-statistic with bootstrap
        normal_samples = np.random.normal(size=[basis.shape[1], n_rep_boot])
        bootstrap_samples = np.multiply(basis.dot(np.dot(sqrtm(blp_omega), normal_samples)).T, (1.0 / blp_se))

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
