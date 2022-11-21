import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.model_selection import train_test_split, KFold
from scipy.optimize import root_scalar

from ._utils_boot import boot_manual, draw_weights
from ._utils import fit_predict_proba

def fit_pq(y, x, d, quantile,
           learner_g, learner_m, all_smpls, treatment, dml_procedure, n_rep=1,
           trimming_threshold=1e-12):
    n_obs = len(y)

    pqs = np.zeros(n_rep)
    ses = np.zeros(n_rep)

    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]

        g_hat, m_hat = fit_nuisance_pq(y, x, d, quantile,
                                       learner_g, learner_m, smpls, treatment,
                                       trimming_threshold=trimming_threshold)

        if dml_procedure == 'dml1':
            pqs[i_rep], ses[i_rep] = pq_dml1(y, d,
                                            g_hat, m_hat, treatment, quantile, smpls)
        else:
            pqs[i_rep], ses[i_rep] = pq_dml2(y, d,
                                            g_hat, m_hat, treatment, quantile)

    pq = np.median(pqs)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(pqs - pq, 2)) / n_obs)

    res = {'pq': pq, 'se': se,
           'pqs': pqs, 'ses': ses}

    return res

def fit_nuisance_pq(y, x, d, quantile, learner_g, learner_m, smpls, treatment, trimming_threshold):
    n_folds = len(smpls)
    n_obs = len(y)

    ml_g = clone(learner_g)
    ml_m = clone(learner_m)
    ml_m_prelim = clone(learner_m)

    # initialize nuisance predictions
    g_hat = np.full(shape=(n_obs), fill_value=np.nan)
    m_hat = np.full(shape=(n_obs), fill_value=np.nan)

    for i_fold in range(len(smpls)):
        train_inds = smpls[i_fold][0]
        test_inds = smpls[i_fold][1]

        # start nested crossfitting
        train_inds_1, train_inds_2 = train_test_split(train_inds, test_size=0.5)
        smpls_prelim = [(train, test) for train, test in KFold(n_splits=n_folds).split(train_inds_1)]

        d_train_1 = d[train_inds_1]
        y_train_1 = y[train_inds_1]
        x_train_1 = x[train_inds_1, :]
        m_hat_prelim = fit_predict_proba(d_train_1, x_train_1, ml_m_prelim, params=None,
                                         trimming_threshold=trimming_threshold,
                                         smpls=smpls_prelim)

        if treatment == 0:
            m_hat_prelim = 1 - m_hat_prelim

        def ipw_score(theta):
            res = (d == treatment) * (y <= theta) / m_hat_prelim - quantile
            return res

        bracket_guess = (y.min(), y.max())
        root_res = root_scalar(ipw_score,
                               bracket=bracket_guess,
                               method='brentq')
        ipw_est = root_res.root

        # use the preliminary estimates to fit the nuisance parameters on train_2
        d_train_2 = d[train_inds_2]
        y_train_2 = y[train_inds_2]
        x_train_2 = x[train_inds_2, :]

        ml_g.fit(np.column_stack((d_train_2[d_train_2 == treatment], x_train_2[d_train_2 == treatment, :])),
                                  y_train_2[d_train_2 == treatment] <= ipw_est)

        # predict nuisance values on the test data
        if treatment == 0:
            dx_test = np.column_stack((np.zeros_like(d[test_inds]), x[test_inds, :]))
        elif treatment == 1:
            dx_test = np.column_stack((np.ones_like(d[test_inds]), x[test_inds, :]))

        g_hat[test_inds] = ml_g.predict_proba(dx_test)[:, 1]

        # refit the propensity score on the whole training set
        ml_m.fit(x[train_inds, :], d[train_inds])
        m_hat[test_inds] = ml_m.predict_proba(x[test_inds, :])[:, treatment]

        m_hat[m_hat < trimming_threshold] = trimming_threshold
        m_hat[m_hat > 1 - trimming_threshold] = 1 - trimming_threshold

        return g_hat, m_hat


def pq_dml1(y, d, g_hat, m_hat, treatment, quantile, smpls):
    thetas = np.zeros(len(smpls))
    n_obs = len(y)
    for idx, (_, test_index) in enumerate(smpls):
        thetas[idx] = pq_est(g_hat[test_index], m_hat[test_index],
                             d[test_index], y[test_index], treatment, quantile)
    theta_hat = np.mean(thetas)

    se = np.sqrt(pq_var_est(theta_hat, g_hat, m_hat, d, y, treatment, quantile, n_obs))

    return theta_hat, se


def pq_dml2(y, d, g_hat, m_hat, treatment, quantile):
    n_obs = len(y)
    theta_hat = pq_est(g_hat, m_hat, d, y, treatment, quantile)

    se = np.sqrt(pq_var_est(theta_hat, g_hat, m_hat, d, y, treatment, quantile, n_obs))

    return theta_hat, se


def pq_est(g_hat, m_hat, d, y, treatment, quantile):
    def compute_score(coef):
        score = (d == treatment) * ((y <= coef) - g_hat) / m_hat + g_hat - quantile
        return np.mean(score)

    bracket_guess = (y.min(), y.max())
    root_res = root_scalar(compute_score,
                           bracket=bracket_guess,
                           method='brentq')
    dml_est = root_res.root

    return dml_est

def pq_var_est(coef, g_hat, m_hat, d, y, treatment, quantile, n_obs, normalize=True, h=None):
    score_weights = (d == treatment) / m_hat
    normalization = score_weights.mean()

    if normalize:
        score_weights /= normalization
    if h is None:
        h = np.power(n_obs, -0.2)
    u = (y - coef).reshape(-1, 1) / h
    kernel_est = np.exp(-1. * np.power(u, 2) / 2) / np.sqrt(2 * np.pi)

    deriv = np.multiply(score_weights, kernel_est.reshape(-1,)) / h
    J = np.mean(deriv)
    score = np.mean((d == treatment) * ((y <= coef) - g_hat) / m_hat + g_hat - quantile)
    var_est = 1/n_obs * np.square(score) / np.square(J)
    return var_est




