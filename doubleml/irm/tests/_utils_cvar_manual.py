import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold

from ...tests._utils import fit_predict_proba, tune_grid_search
from ...utils._estimation import _dml_cv_predict, _normalize_ipw, _get_bracket_guess, _solve_ipw_score


def fit_cvar(y, x, d, quantile,
             learner_g, learner_m, all_smpls, treatment, normalize_ipw=True, n_rep=1,
             trimming_threshold=1e-2, g_params=None, m_params=None):
    n_obs = len(y)

    cvars = np.zeros(n_rep)
    ses = np.zeros(n_rep)

    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]

        g_hat, m_hat, ipw_est = fit_nuisance_cvar(y, x, d, quantile,
                                                  learner_g, learner_m, smpls, treatment,
                                                  normalize_ipw=normalize_ipw,
                                                  trimming_threshold=trimming_threshold,
                                                  g_params=g_params, m_params=m_params)

        cvars[i_rep], ses[i_rep] = cvar_dml2(y, d, g_hat, m_hat, treatment, quantile, ipw_est)

    cvar = np.median(cvars)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(cvars - cvar, 2)) / n_obs)

    res = {'pq': cvar, 'se': se,
           'pqs': cvars, 'ses': ses}

    return res


def fit_nuisance_cvar(y, x, d, quantile, learner_g, learner_m, smpls, treatment,
                      normalize_ipw, trimming_threshold, g_params, m_params):
    n_folds = len(smpls)
    n_obs = len(y)
    coef_bounds = (y.min(), y.max())
    y_treat = y[d == treatment]
    coef_start_val = np.mean(y_treat[y_treat >= np.quantile(y_treat, quantile)])

    ml_g = clone(learner_g)
    ml_m = clone(learner_m)

    # initialize nuisance predictions
    g_hat = np.full(shape=n_obs, fill_value=np.nan)
    m_hat = np.full(shape=n_obs, fill_value=np.nan)

    ipw_vec = np.full(shape=n_folds, fill_value=np.nan)
    for i_fold, _ in enumerate(smpls):
        ml_g = clone(learner_g)
        ml_m = clone(learner_m)
        # set the params for the nuisance learners
        if g_params is not None:
            ml_g.set_params(**g_params[i_fold])
        if m_params is not None:
            ml_m.set_params(**m_params[i_fold])

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

        m_hat_prelim[m_hat_prelim < trimming_threshold] = trimming_threshold
        m_hat_prelim[m_hat_prelim > 1 - trimming_threshold] = 1 - trimming_threshold

        if normalize_ipw:
            m_hat_prelim = _normalize_ipw(m_hat_prelim, d_train_1)
        if treatment == 0:
            m_hat_prelim = 1 - m_hat_prelim

        def ipw_score(theta):
            res = np.mean((d_train_1 == treatment) * (y_train_1 <= theta) / m_hat_prelim - quantile)
            return res

        _, bracket_guess = _get_bracket_guess(ipw_score, coef_start_val, coef_bounds)
        ipw_est = _solve_ipw_score(ipw_score=ipw_score, bracket_guess=bracket_guess)
        ipw_vec[i_fold] = ipw_est

        # use the preliminary estimates to fit the nuisance parameters on train_2
        d_train_2 = d[train_inds_2]
        x_train_2 = x[train_inds_2, :]

        # calculate the target for g
        g_target_1 = np.ones_like(y) * ipw_est
        g_target_2 = (y - quantile * ipw_est) / (1 - quantile)
        g_target = np.max(np.column_stack((g_target_1, g_target_2)), 1)
        g_target_train_2 = g_target[train_inds_2]

        dx_treat_train_2 = x_train_2[d_train_2 == treatment, :]
        g_target_train_2_d = g_target_train_2[d_train_2 == treatment]
        ml_g.fit(dx_treat_train_2, g_target_train_2_d)

        # predict nuisance values on the test data
        g_hat[test_inds] = ml_g.predict(x[test_inds, :])

        # refit the propensity score on the whole training set
        ml_m.fit(x[train_inds, :], d[train_inds])
        m_hat[test_inds] = ml_m.predict_proba(x[test_inds, :])[:, 1]

    m_hat[m_hat < trimming_threshold] = trimming_threshold
    m_hat[m_hat > 1 - trimming_threshold] = 1 - trimming_threshold

    if normalize_ipw:
        m_hat = _normalize_ipw(m_hat, d)

    if treatment == 0:
        m_hat = 1 - m_hat

    ipw_est = np.mean(ipw_vec)
    return g_hat, m_hat, ipw_est


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


def tune_nuisance_cvar(y, x, d, ml_g, ml_m, smpls, treatment, quantile, n_folds_tune,
                       param_grid_g, param_grid_m):
    train_cond_treat = np.where(d == treatment)[0]

    quantile_approx = np.quantile(y[d == treatment], quantile)
    g_target_1 = np.ones_like(y) * quantile_approx
    g_target_2 = (y - quantile * quantile_approx) / (1 - quantile)
    g_target_approx = np.max(np.column_stack((g_target_1, g_target_2)), 1)
    g_tune_res = tune_grid_search(g_target_approx, x, ml_g, smpls, param_grid_g, n_folds_tune,
                                  train_cond=train_cond_treat)
    m_tune_res = tune_grid_search(d, x, ml_m, smpls, param_grid_m, n_folds_tune)

    g_best_params = [xx.best_params_ for xx in g_tune_res]
    m_best_params = [xx.best_params_ for xx in m_tune_res]

    return g_best_params, m_best_params
