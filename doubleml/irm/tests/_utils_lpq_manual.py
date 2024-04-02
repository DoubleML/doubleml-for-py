import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy.optimize import root_scalar

from ...tests._utils import tune_grid_search
from ...utils._estimation import _dml_cv_predict, _trimm, _default_kde, _normalize_ipw, _get_bracket_guess, _solve_ipw_score


def fit_lpq(y, x, d, z, quantile,
            learner_g, learner_m, all_smpls, treatment, n_rep=1,
            trimming_rule='truncate',
            trimming_threshold=1e-2,
            kde=_default_kde,
            normalize_ipw=True, m_z_params=None,
            m_d_z0_params=None, m_d_z1_params=None,
            g_du_z0_params=None, g_du_z1_params=None):
    n_obs = len(y)

    lpqs = np.zeros(n_rep)
    ses = np.zeros(n_rep)

    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]

        m_z_hat, g_du_z0_hat, g_du_z1_hat, \
            comp_prob_hat, ipw_vec, coef_bounds = fit_nuisance_lpq(y, x, d, z, quantile,
                                                                   learner_g, learner_m, smpls,
                                                                   treatment,
                                                                   trimming_rule=trimming_rule,
                                                                   trimming_threshold=trimming_threshold,
                                                                   normalize_ipw=normalize_ipw,
                                                                   m_z_params=m_z_params,
                                                                   m_d_z0_params=m_d_z0_params,
                                                                   m_d_z1_params=m_d_z1_params,
                                                                   g_du_z0_params=g_du_z0_params,
                                                                   g_du_z1_params=g_du_z1_params)

        lpqs[i_rep], ses[i_rep] = lpq_dml2(y, d, z, m_z_hat, g_du_z0_hat, g_du_z1_hat, comp_prob_hat,
                                           treatment, quantile, ipw_vec, coef_bounds, kde)

    lpq = np.median(lpqs)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(lpqs - lpq, 2)) / n_obs)

    res = {'lpq': lpq, 'se': se,
           'lpqs': lpqs, 'ses': ses}

    return res


def fit_nuisance_lpq(y, x, d, z, quantile, learner_g, learner_m, smpls, treatment,
                     trimming_rule, trimming_threshold, normalize_ipw, m_z_params,
                     m_d_z0_params, m_d_z1_params, g_du_z0_params, g_du_z1_params):
    n_folds = len(smpls)
    n_obs = len(y)
    # initialize starting values and bounds
    coef_bounds = (y.min(), y.max())
    y_treat = y[d == treatment]
    coef_start_val = np.quantile(y_treat, quantile)

    strata = d + 2 * z

    # initialize nuisance predictions
    m_z_hat = np.full(shape=n_obs, fill_value=np.nan)
    m_d_z0_hat = np.full(shape=n_obs, fill_value=np.nan)
    m_d_z1_hat = np.full(shape=n_obs, fill_value=np.nan)
    g_du_z0_hat = np.full(shape=n_obs, fill_value=np.nan)
    g_du_z1_hat = np.full(shape=n_obs, fill_value=np.nan)

    ipw_vec = np.full(shape=n_folds, fill_value=np.nan)
    for i_fold, _ in enumerate(smpls):
        ml_m_z = clone(learner_m)
        ml_m_d_z0 = clone(learner_m)
        ml_m_d_z1 = clone(learner_m)
        ml_g_du_z0 = clone(learner_g)
        ml_g_du_z1 = clone(learner_g)
        # set the params for the nuisance learners
        if m_z_params is not None:
            ml_m_z.set_params(**m_z_params[i_fold])
        if m_d_z0_params is not None:
            ml_m_d_z0.set_params(**m_d_z0_params[i_fold])
        if m_d_z1_params is not None:
            ml_m_d_z1.set_params(**m_d_z1_params[i_fold])
        if g_du_z0_params is not None:
            ml_g_du_z0.set_params(**g_du_z0_params[i_fold])
        if g_du_z1_params is not None:
            ml_g_du_z1.set_params(**g_du_z1_params[i_fold])

        train_inds = smpls[i_fold][0]
        test_inds = smpls[i_fold][1]

        # start nested crossfitting
        train_inds_1, train_inds_2 = train_test_split(train_inds, test_size=0.5,
                                                      random_state=42, stratify=strata[train_inds])
        smpls_prelim = [(train, test) for train, test in
                        StratifiedKFold(n_splits=n_folds).split(X=train_inds_1, y=strata[train_inds_1])]

        d_train_1 = d[train_inds_1]
        y_train_1 = y[train_inds_1]
        x_train_1 = x[train_inds_1, :]
        z_train_1 = z[train_inds_1]

        # preliminary propensity for z
        # todo change prediction method
        ml_m_z_prelim = clone(ml_m_z)
        m_z_hat_prelim = _dml_cv_predict(ml_m_z_prelim, x_train_1, z_train_1,
                                         method='predict_proba', smpls=smpls_prelim)['preds']

        m_z_hat_prelim = _trimm(m_z_hat_prelim, trimming_rule, trimming_threshold)
        if normalize_ipw:
            m_z_hat_prelim = _normalize_ipw(m_z_hat_prelim, z_train_1)

        # propensity for d == 1 cond. on z == 0 (training set 1)
        x_z0_train_1 = x_train_1[z_train_1 == 0, :]
        d_z0_train_1 = d_train_1[z_train_1 == 0]
        ml_m_d_z0_prelim = clone(ml_m_d_z0)
        ml_m_d_z0_prelim.fit(x_z0_train_1, d_z0_train_1)
        m_d_z0_hat_prelim = ml_m_d_z0_prelim.predict_proba(x_train_1)[:, 1]

        # propensity for d == 1 cond. on z == 1 (training set 1)
        x_z1_train_1 = x_train_1[z_train_1 == 1, :]
        d_z1_train_1 = d_train_1[z_train_1 == 1]
        ml_m_d_z1_prelim = clone(ml_m_d_z1)
        ml_m_d_z1_prelim.fit(x_z1_train_1, d_z1_train_1)
        m_d_z1_hat_prelim = ml_m_d_z1_prelim.predict_proba(x_train_1)[:, 1]

        # preliminary estimate of theta_2_aux
        comp_prob_prelim = np.mean(m_d_z1_hat_prelim - m_d_z0_hat_prelim
                                   + z_train_1 / m_z_hat_prelim * (d_train_1 - m_d_z1_hat_prelim)
                                   - (1 - z_train_1) / (1 - m_z_hat_prelim) * (d_train_1 - m_d_z0_hat_prelim))

        def ipw_score(theta):
            sign = 2 * treatment - 1.0
            weights = sign * (z_train_1 / m_z_hat_prelim - (1 - z_train_1) / (1 - m_z_hat_prelim)) / comp_prob_prelim
            u = (d_train_1 == treatment) * (y_train_1 <= theta)
            v = -1. * quantile
            res = np.mean(weights * u + v)
            return res

        _, bracket_guess = _get_bracket_guess(ipw_score, coef_start_val, coef_bounds)
        ipw_est = _solve_ipw_score(ipw_score=ipw_score, bracket_guess=bracket_guess)
        ipw_vec[i_fold] = ipw_est

        # use the preliminary estimates to fit the nuisance parameters on train_2
        d_train_2 = d[train_inds_2]
        y_train_2 = y[train_inds_2]
        x_train_2 = x[train_inds_2, :]
        z_train_2 = z[train_inds_2]

        # propensity for (D == treatment)*Ind(Y <= ipq_est) cond. on z == 0
        x_z0_train_2 = x_train_2[z_train_2 == 0, :]
        du_z0_train_2 = (d_train_2[z_train_2 == 0] == treatment) * (y_train_2[z_train_2 == 0] <= ipw_est)
        ml_g_du_z0.fit(x_z0_train_2, du_z0_train_2)
        g_du_z0_hat[test_inds] = ml_g_du_z0.predict_proba(x[test_inds, :])[:, 1]

        # propensity for (D == treatment)*Ind(Y <= ipq_est) cond. on z == 1
        x_z1_train_2 = x_train_2[z_train_2 == 1, :]
        du_z1_train_2 = (d_train_2[z_train_2 == 1] == treatment) * (y_train_2[z_train_2 == 1] <= ipw_est)
        ml_g_du_z1.fit(x_z1_train_2, du_z1_train_2)
        g_du_z1_hat[test_inds] = ml_g_du_z1.predict_proba(x[test_inds, :])[:, 1]

        # refit nuisance elements for the local potential quantile
        z_train = z[train_inds]
        x_train = x[train_inds]
        d_train = d[train_inds]

        # refit propensity for z (whole training set)
        ml_m_z.fit(x_train, z_train)
        m_z_hat[test_inds] = ml_m_z.predict_proba(x[test_inds, :])[:, 1]

        # refit propensity for d == 1 cond. on z == 0 (whole training set)
        x_z0_train = x_train[z_train == 0, :]
        d_z0_train = d_train[z_train == 0]
        ml_m_d_z0.fit(x_z0_train, d_z0_train)
        m_d_z0_hat[test_inds] = ml_m_d_z0.predict_proba(x[test_inds, :])[:, 1]

        # propensity for d == 1 cond. on z == 1 (whole training set)
        x_z1_train = x_train[z_train == 1, :]
        d_z1_train = d_train[z_train == 1]
        ml_m_d_z1.fit(x_z1_train, d_z1_train)
        m_d_z1_hat[test_inds] = ml_m_d_z1.predict_proba(x[test_inds, :])[:, 1]

    # clip propensities
    m_z_hat = _trimm(m_z_hat, trimming_rule, trimming_threshold)

    if normalize_ipw:
        m_z_hat = _normalize_ipw(m_z_hat, z)

    # estimate final nuisance parameter
    comp_prob_hat = np.mean(m_d_z1_hat - m_d_z0_hat
                            + z / m_z_hat * (d - m_d_z1_hat)
                            - (1 - z) / (1 - m_z_hat) * (d - m_d_z0_hat))
    return m_z_hat, g_du_z0_hat, g_du_z1_hat, comp_prob_hat, ipw_vec, coef_bounds


def lpq_dml2(y, d, z, m_z, g_du_z0, g_du_z1, comp_prob, treatment, quantile, ipw_vec, coef_bounds, kde):
    n_obs = len(y)
    ipw_est = ipw_vec.mean()
    theta_hat = lpq_est(m_z, g_du_z0, g_du_z1, comp_prob, d, y, z, treatment, quantile, ipw_est, coef_bounds)

    se = np.sqrt(lpq_var_est(theta_hat, m_z, g_du_z0, g_du_z1, comp_prob, d, y, z, treatment, quantile, n_obs, kde))

    return theta_hat, se


def lpq_est(m_z, g_du_z0, g_du_z1, comp_prob, d, y, z, treatment, quantile, ipw_est, coef_bounds):

    def compute_score(coef):
        sign = 2 * treatment - 1.0
        score1 = g_du_z1 - g_du_z0
        score2 = (z / m_z) * ((d == treatment) * (y <= coef) - g_du_z1)
        score3 = (1 - z) / (1 - m_z) * ((d == treatment) * (y <= coef) - g_du_z0)
        score = sign * (score1 + score2 - score3) / comp_prob - quantile
        return np.mean(score)

    def compute_score_mean(coef):
        return np.mean(compute_score(coef))

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

    _, bracket_guess = get_bracket_guess(ipw_est, coef_bounds)
    root_res = root_scalar(compute_score_mean,
                           bracket=bracket_guess,
                           method='brentq')
    dml_est = root_res.root
    return dml_est


def lpq_var_est(coef, m_z, g_du_z0, g_du_z1, comp_prob, d, y, z, treatment, quantile, n_obs, kde=_default_kde):
    sign = 2 * treatment - 1.0
    score_weights = sign * ((z / m_z) - (1 - z) / (1 - m_z)) * (d == treatment) / comp_prob
    u = (y - coef).reshape(-1, 1)
    deriv = kde(u, score_weights)

    J = np.mean(deriv)
    sign = 2 * treatment - 1.0
    score1 = g_du_z1 - g_du_z0
    score2 = (z / m_z) * ((d == treatment) * (y <= coef) - g_du_z1)
    score3 = (1 - z) / (1 - m_z) * ((d == treatment) * (y <= coef) - g_du_z0)
    score = sign * (score1 + score2 - score3) / comp_prob - quantile
    var_est = 1/n_obs * np.mean(np.square(score)) / np.square(J)
    return var_est


def tune_nuisance_lpq(y, x, d, z,
                      ml_m_z, ml_m_d_z0, ml_m_d_z1, ml_g_du_z0, ml_g_du_z1,
                      smpls, treatment, quantile, n_folds_tune,
                      param_grid_m_z, param_grid_m_d_z0, param_grid_m_d_z1,
                      param_grid_g_du_z0, param_grid_g_du_z1):
    train_cond_z0 = np.where(z == 0)[0]
    train_cond_z1 = np.where(z == 1)[0]

    approx_quant = np.quantile(y[d == treatment], quantile)
    du = (d == treatment) * (y <= approx_quant)

    m_z_tune_res = tune_grid_search(z, x, ml_m_z, smpls, param_grid_m_z, n_folds_tune)
    m_d_z0_tune_res = tune_grid_search(d, x, ml_m_d_z0, smpls, param_grid_m_d_z0, n_folds_tune,
                                       train_cond=train_cond_z0)
    m_d_z1_tune_res = tune_grid_search(d, x, ml_m_d_z1, smpls, param_grid_m_d_z1, n_folds_tune,
                                       train_cond=train_cond_z1)
    g_du_z0_tune_res = tune_grid_search(du, x, ml_g_du_z0, smpls, param_grid_g_du_z0, n_folds_tune,
                                        train_cond=train_cond_z0)
    g_du_z1_tune_res = tune_grid_search(du, x, ml_g_du_z1, smpls, param_grid_g_du_z1, n_folds_tune,
                                        train_cond=train_cond_z1)

    m_z_best_params = [xx.best_params_ for xx in m_z_tune_res]
    m_d_z0_best_params = [xx.best_params_ for xx in m_d_z0_tune_res]
    m_d_z1_best_params = [xx.best_params_ for xx in m_d_z1_tune_res]
    g_du_z0_best_params = [xx.best_params_ for xx in g_du_z0_tune_res]
    g_du_z1_best_params = [xx.best_params_ for xx in g_du_z1_tune_res]

    return m_z_best_params, m_d_z0_best_params, m_d_z1_best_params, g_du_z0_best_params, g_du_z1_best_params
