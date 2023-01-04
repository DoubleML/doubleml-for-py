import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy.optimize import root_scalar

from ._utils import fit_predict_proba
from .._utils import _dml_cv_predict


def fit_cvar(y, x, d, quantile,
             learner_g, learner_m, all_smpls, treatment, dml_procedure, n_rep=1,
             trimming_threshold=1e-12):
    n_obs = len(y)

    cvars = np.zeros(n_rep)
    ses = np.zeros(n_rep)

    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]

        g_hat, m_hat, ipw_est = fit_nuisance_cvar(y, x, d, quantile,
                                                  learner_g, learner_m, smpls, treatment,
                                                  trimming_threshold=trimming_threshold)

        if dml_procedure == 'dml1':
            cvars[i_rep], ses[i_rep] = cvar_dml1(y, d, g_hat, m_hat, treatment, quantile, smpls, ipw_est)
        else:
            cvars[i_rep], ses[i_rep] = cvar_dml2(y, d, g_hat, m_hat, treatment, quantile, ipw_est)

    cvar = np.median(cvars)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(cvars - cvar, 2)) / n_obs)

    res = {'pq': cvar, 'se': se,
           'pqs': cvars, 'ses': ses}

    return res


def fit_nuisance_cvar(y, x, d, quantile, learner_g, learner_m, smpls, treatment, trimming_threshold):
    n_folds = len(smpls)
    n_obs = len(y)

    ml_g = clone(learner_g)
    ml_m = clone(learner_m)

    # initialize nuisance predictions
    g_hat = np.full(shape=n_obs, fill_value=np.nan)
    m_hat = np.full(shape=n_obs, fill_value=np.nan)

    for i_fold, _ in enumerate(smpls):
        train_inds = smpls[i_fold][0]
        test_inds = smpls[i_fold][1]

        # start nested crossfitting
        train_inds_1, train_inds_2 = train_test_split(train_inds, test_size=0.5,
                                                      random_state=42, stratify=d[train_inds])
        smpls_prelim = [(train, test) for train, test in
                        StratifiedKFold(n_splits=n_folds).split(X=train_inds_1, y=d[train_inds_1])]

        d_train_1 = d[train_inds_1]
        y_train_1 = y[train_inds_1]
        x_train_1 = x[train_inds_1, :]
        # todo change prediction method
        m_hat_prelim_list = fit_predict_proba(d_train_1, x_train_1, ml_m,
                                              params=None,
                                              trimming_threshold=trimming_threshold,
                                              smpls=smpls_prelim)

        m_hat_prelim = np.full_like(y_train_1, np.nan, dtype='float64')
        for idx, (_, test_index) in enumerate(smpls_prelim):
            m_hat_prelim[test_index] = m_hat_prelim_list[idx]

        m_hat_prelim = _dml_cv_predict(ml_m, x_train_1, d_train_1,
                                       method='predict_proba', smpls=smpls_prelim)['preds']
        if treatment == 0:
            m_hat_prelim = 1 - m_hat_prelim

        def ipw_score(theta):
            res = np.mean((d_train_1 == treatment) * (y_train_1 <= theta) / m_hat_prelim - quantile)
            return res

        def get_bracket_guess(coef_start, coef_bounds):
            max_bracket_length = coef_bounds[1] - coef_bounds[0]
            b_guess = coef_bounds
            delta = 0.1
            s_different = False
            while (not s_different) & (delta <= 1.0):
                a = np.maximum(coef_start - delta * max_bracket_length / 2, coef_bounds[0])
                b = np.minimum(coef_start + delta * max_bracket_length / 2, coef_bounds[1])
                b_guess = (a, b)
                f_a = ipw_score(b_guess[0])
                f_b = ipw_score(b_guess[1])
                s_different = (np.sign(f_a) != np.sign(f_b))
                delta += 0.1
            return s_different, b_guess

        coef_start_val = np.quantile(y, q=quantile)
        coef_bounds = (y.min(), y.max())
        _, bracket_guess = get_bracket_guess(coef_start_val, coef_bounds)

        root_res = root_scalar(ipw_score,
                               bracket=bracket_guess,
                               method='brentq')
        ipw_est = root_res.root

        # use the preliminary estimates to fit the nuisance parameters on train_2
        d_train_2 = d[train_inds_2]
        y_train_2 = y[train_inds_2]
        x_train_2 = x[train_inds_2, :]

        dx_treat_train_2 = x_train_2[d_train_2 == treatment, :]
        y_treat_train_2 = y_train_2[d_train_2 == treatment]
        ml_g.fit(dx_treat_train_2, y_treat_train_2 <= ipw_est)

        # predict nuisance values on the test data
        g_hat[test_inds] = ml_g.predict_proba(x[test_inds, :])[:, 1]

        # refit the propensity score on the whole training set
        ml_m.fit(x[train_inds, :], d[train_inds])
        m_hat[test_inds] = ml_m.predict_proba(x[test_inds, :])[:, treatment]

    m_hat[m_hat < trimming_threshold] = trimming_threshold
    m_hat[m_hat > 1 - trimming_threshold] = 1 - trimming_threshold

    return g_hat, m_hat, ipw_est


def cvar_dml1(y, d, g_hat, m_hat, treatment, quantile, smpls, ipw_est):
    thetas = np.zeros(len(smpls))
    n_obs = len(y)
    for idx, (_, test_index) in enumerate(smpls):
        thetas[idx] = cvar_est(g_hat[test_index], m_hat[test_index],
                               d[test_index], y[test_index], treatment, quantile, ipw_est)
    theta_hat = np.mean(thetas)

    se = np.sqrt(cvar_var_est(theta_hat, g_hat, m_hat, d, y, treatment, quantile, ipw_est, n_obs))

    return theta_hat, se


def cvar_dml2(y, d, g_hat, m_hat, treatment, quantile, ipw_est):
    n_obs = len(y)
    theta_hat = cvar_est(g_hat, m_hat, d, y, treatment, quantile, ipw_est)

    se = np.sqrt(cvar_var_est(theta_hat, g_hat, m_hat, d, y, treatment, quantile, ipw_est, n_obs))

    return theta_hat, se


def cvar_est(g_hat, m_hat, d, y, treatment, quantile, ipw_est):
    u1 = np.ones_like(y) * ipw_est
    u2 = (y - quantile * ipw_est) / (1 - quantile)
    u = np.max(np.column_stack((u1, u2)), 1)

    psi_b = (d == treatment) * (u - g_hat) / m_hat + g_hat
    psi_a = np.full_like(m_hat, -1.0)
    dml_est = -np.mean(psi_b) / np.mean(psi_a)
    return dml_est


def cvar_var_est(coef, g_hat, m_hat, d, y, treatment, quantile, ipw_est, n_obs):
    u1 = np.ones_like(y) * ipw_est
    u2 = (y - quantile * ipw_est) / (1 - quantile)
    u = np.max(np.column_stack((u1, u2)), 1)

    psi_a = np.full_like(m_hat, -1.0)
    psi_b = (d == treatment) * (u - g_hat) / m_hat + g_hat

    J = np.mean(psi_a)
    psi = psi_a * coef + psi_b
    var_est = 1 / n_obs * np.mean(np.power(psi, 2)) / np.power(J, 2)
    return var_est
