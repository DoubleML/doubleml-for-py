import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy.optimize import root_scalar

from ...tests._utils import tune_grid_search
from ...utils._estimation import _dml_cv_predict, _default_kde, _normalize_ipw, _solve_ipw_score, _get_bracket_guess


def fit_pq(y, x, d, quantile,
           learner_g, learner_m, all_smpls, treatment, n_rep=1,
           trimming_threshold=1e-2, normalize_ipw=True, g_params=None, m_params=None):
    n_obs = len(y)

    pqs = np.zeros(n_rep)
    ses = np.zeros(n_rep)

    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]

        g_hat, m_hat, ipw_est = fit_nuisance_pq(y, x, d, quantile,
                                                learner_g, learner_m, smpls, treatment,
                                                trimming_threshold=trimming_threshold,
                                                normalize_ipw=normalize_ipw,
                                                g_params=g_params, m_params=m_params)

        pqs[i_rep], ses[i_rep] = pq_dml2(y, d, g_hat, m_hat, treatment, quantile, ipw_est)

    pq = np.median(pqs)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(pqs - pq, 2)) / n_obs)

    res = {'pq': pq, 'se': se,
           'pqs': pqs, 'ses': ses}

    return res


def fit_nuisance_pq(y, x, d, quantile, learner_g, learner_m, smpls, treatment,
                    trimming_threshold, normalize_ipw, g_params, m_params):
    n_folds = len(smpls)
    n_obs = len(y)
    # initialize starting values and bounds
    coef_bounds = (y.min(), y.max())
    y_treat = y[d == treatment]
    coef_start_val = np.quantile(y_treat, quantile)

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
        m_hat_prelim = _dml_cv_predict(clone(ml_m), x_train_1, d_train_1,
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
        y_train_2 = y[train_inds_2]
        x_train_2 = x[train_inds_2, :]

        dx_treat_train_2 = x_train_2[d_train_2 == treatment, :]
        y_treat_train_2 = y_train_2[d_train_2 == treatment]
        ml_g.fit(dx_treat_train_2, y_treat_train_2 <= ipw_est)

        # predict nuisance values on the test data
        g_hat[test_inds] = ml_g.predict_proba(x[test_inds, :])[:, 1]

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


def pq_dml2(y, d, g_hat, m_hat, treatment, quantile, ipw_est):
    n_obs = len(y)
    theta_hat = pq_est(g_hat, m_hat, d, y, treatment, quantile, ipw_est)

    se = np.sqrt(pq_var_est(theta_hat, g_hat, m_hat, d, y, treatment, quantile, n_obs))

    return theta_hat, se


def pq_est(g_hat, m_hat, d, y, treatment, quantile, ipw_est):
    def compute_score(coef):
        score = (d == treatment) * ((y <= coef) - g_hat) / m_hat + g_hat - quantile
        return np.mean(score)

    def get_bracket_guess(coef_start, coef_bounds):
        max_bracket_length = coef_bounds[1] - coef_bounds[0]
        b_guess = coef_bounds
        delta = 0.1
        s_different = False
        while (not s_different) & (delta <= 1.0):
            a = np.maximum(coef_start - delta * max_bracket_length / 2, coef_bounds[0])
            b = np.minimum(coef_start + delta * max_bracket_length / 2, coef_bounds[1])
            b_guess = (a, b)
            f_a = compute_score(b_guess[0])
            f_b = compute_score(b_guess[1])
            s_different = (np.sign(f_a) != np.sign(f_b))
            delta += 0.1
        return s_different, b_guess

    coef_start_val = ipw_est
    coef_bounds = (y.min(), y.max())
    _, bracket_guess = get_bracket_guess(coef_start_val, coef_bounds)

    root_res = root_scalar(compute_score,
                           bracket=bracket_guess,
                           method='brentq')
    dml_est = root_res.root

    return dml_est


def pq_var_est(coef, g_hat, m_hat, d, y, treatment, quantile, n_obs, kde=_default_kde):
    score_weights = (d == treatment) / m_hat
    u = (y - coef).reshape(-1, 1)
    deriv = kde(u, score_weights)

    J = np.mean(deriv)
    score = (d == treatment) * ((y <= coef) - g_hat) / m_hat + g_hat - quantile
    var_est = 1/n_obs * np.mean(np.square(score)) / np.square(J)
    return var_est


def tune_nuisance_pq(y, x, d, ml_g, ml_m, smpls, treatment, quantile, n_folds_tune,
                     param_grid_g, param_grid_m):
    train_cond_treat = np.where(d == treatment)[0]
    approx_goal = y <= np.quantile(y[d == treatment], quantile)
    g_tune_res = tune_grid_search(approx_goal, x, ml_g, smpls, param_grid_g, n_folds_tune,
                                  train_cond=train_cond_treat)
    m_tune_res = tune_grid_search(d, x, ml_m, smpls, param_grid_m, n_folds_tune)

    g_best_params = [xx.best_params_ for xx in g_tune_res]
    m_best_params = [xx.best_params_ for xx in m_tune_res]

    return g_best_params, m_best_params
