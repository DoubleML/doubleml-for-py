import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy.optimize import root_scalar

from .._utils import _dml_cv_predict, _default_kde, _normalize_ipw


def fit_pq(y, x, d, quantile,
           learner_g, learner_m, all_smpls, treatment, dml_procedure, n_rep=1,
           trimming_threshold=1e-2, normalize_ipw=True):
    n_obs = len(y)

    pqs = np.zeros(n_rep)
    ses = np.zeros(n_rep)

    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]

        g_hat, m_hat, ipw_est = fit_nuisance_pq(y, x, d, quantile,
                                                learner_g, learner_m, smpls, treatment,
                                                trimming_threshold=trimming_threshold,
                                                normalize_ipw=normalize_ipw,
                                                dml_procedure=dml_procedure)

        if dml_procedure == 'dml1':
            pqs[i_rep], ses[i_rep] = pq_dml1(y, d, g_hat, m_hat, treatment, quantile, smpls, ipw_est)
        else:
            pqs[i_rep], ses[i_rep] = pq_dml2(y, d, g_hat, m_hat, treatment, quantile, ipw_est)

    pq = np.median(pqs)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(pqs - pq, 2)) / n_obs)

    res = {'pq': pq, 'se': se,
           'pqs': pqs, 'ses': ses}

    return res


def fit_nuisance_pq(y, x, d, quantile, learner_g, learner_m, smpls, treatment,
                    trimming_threshold, normalize_ipw, dml_procedure):
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
        m_hat[test_inds] = ml_m.predict_proba(x[test_inds, :])[:, 1]

    m_hat[m_hat < trimming_threshold] = trimming_threshold
    m_hat[m_hat > 1 - trimming_threshold] = 1 - trimming_threshold

    if normalize_ipw:
        if dml_procedure == 'dml1':
            for _, test_index in smpls:
                m_hat[test_index] = _normalize_ipw(m_hat[test_index], d[test_index])
        else:
            m_hat = _normalize_ipw(m_hat, d)

    if treatment == 0:
        m_hat = 1 - m_hat

    return g_hat, m_hat, ipw_est


def pq_dml1(y, d, g_hat, m_hat, treatment, quantile, smpls, ipw_est):
    thetas = np.zeros(len(smpls))
    n_obs = len(y)
    for idx, (_, test_index) in enumerate(smpls):
        thetas[idx] = pq_est(g_hat[test_index], m_hat[test_index],
                             d[test_index], y[test_index], treatment, quantile, ipw_est)
    theta_hat = np.mean(thetas)

    se = np.sqrt(pq_var_est(theta_hat, g_hat, m_hat, d, y, treatment, quantile, n_obs))

    return theta_hat, se


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
