import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split, KFold
from scipy.optimize import root_scalar

from ._utils import fit_predict_proba
from .._utils import _dml_cv_predict, _trimm


def fit_lpq(y, x, d, z, quantile,
            learner_m, all_smpls, treatment, dml_procedure, n_rep=1,
            trimming_rule='truncate',
            trimming_threshold=1e-12):
    n_obs = len(y)

    lpqs = np.zeros(n_rep)
    ses = np.zeros(n_rep)

    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]

        pi_z_hat, pi_du_z0_hat, pi_du_z1_hat, comp_prob_hat, ipw_est = fit_nuisance_lpq(y, x,
                                                                                        d, z, quantile,
                                                                                        learner_m, smpls, treatment,
                                                                                        trimming_rule=trimming_rule,
                                                                                        trimming_threshold=trimming_threshold)
        print(f'comp_prob manual: {comp_prob_hat}')
        if dml_procedure == 'dml1':
            lpqs[i_rep], ses[i_rep] = lpq_dml1(y, d, z, pi_z_hat, pi_du_z0_hat, pi_du_z1_hat, comp_prob_hat,
                                               treatment, quantile, ipw_est, smpls)
        else:
            lpqs[i_rep], ses[i_rep] = lpq_dml2(y, d, z, pi_z_hat, pi_du_z0_hat, pi_du_z1_hat, comp_prob_hat,
                                               treatment, quantile, ipw_est)


    lpq = np.median(lpqs)
    print(lpq)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(lpqs - lpq, 2)) / n_obs)

    res = {'lpq': lpq, 'se': se,
           'lpqs': lpqs, 'ses': ses}

    return res


def fit_nuisance_lpq(y, x, d, z, quantile, learner_m, smpls, treatment, trimming_rule, trimming_threshold):
    n_folds = len(smpls)
    n_obs = len(y)

    ml_pi_z = clone(learner_m)
    ml_pi_du_z0 = clone(learner_m)
    ml_pi_du_z1 = clone(learner_m)
    ml_pi_d_z0 = clone(learner_m)
    ml_pi_d_z1 = clone(learner_m)

    # initialize nuisance predictions
    pi_z_hat = np.full(shape=n_obs, fill_value=np.nan)
    pi_d_z0_hat = np.full(shape=n_obs, fill_value=np.nan)
    pi_d_z1_hat = np.full(shape=n_obs, fill_value=np.nan)
    pi_du_z0_hat = np.full(shape=n_obs, fill_value=np.nan)
    pi_du_z1_hat = np.full(shape=n_obs, fill_value=np.nan)

    for i_fold, _ in enumerate(smpls):
        train_inds = smpls[i_fold][0]
        test_inds = smpls[i_fold][1]

        # start nested crossfitting
        train_inds_1, train_inds_2 = train_test_split(train_inds, test_size=0.5, random_state=42)
        smpls_prelim = [(train, test) for train, test in KFold(n_splits=n_folds).split(train_inds_1)]

        d_train_1 = d[train_inds_1]
        y_train_1 = y[train_inds_1]
        x_train_1 = x[train_inds_1, :]
        z_train_1 = z[train_inds_1]

        # preliminary propensity for z
        # todo change prediction method
        pi_z_hat_prelim = _dml_cv_predict(ml_pi_z, x_train_1, z_train_1,
                                          method='predict_proba', smpls=smpls_prelim)['preds']
        pi_z_hat_prelim = _trimm(pi_z_hat_prelim, trimming_rule, trimming_threshold)

        # todo add extra fold loop
        # propensity for d == 1 cond. on z == 0 (training set 1)
        x_z0_train_1 = x_train_1[z_train_1 == 0, :]
        d_z0_train_1 = d_train_1[z_train_1 == 0]
        ml_pi_d_z0.fit(x_z0_train_1, d_z0_train_1)
        pi_d_z0_hat_prelim = ml_pi_d_z0.predict_proba(x_train_1)[:, 1]
        pi_d_z0_hat_prelim = _trimm(pi_d_z0_hat_prelim, trimming_rule, trimming_threshold)

        # propensity for d == 1 cond. on z == 1 (training set 1)
        x_z1_train_1 = x_train_1[z_train_1 == 1, :]
        d_z1_train_1 = d_train_1[z_train_1 == 1]
        ml_pi_d_z1.fit(x_z1_train_1, d_z1_train_1)
        pi_d_z1_hat_prelim = ml_pi_d_z1.predict_proba(x_train_1)[:, 1]
        pi_d_z1_hat_prelim = _trimm(pi_d_z1_hat_prelim, trimming_rule, trimming_threshold)

        # preliminary estimate of theta_2_aux
        comp_prob_prelim = np.mean(pi_d_z1_hat_prelim - pi_d_z0_hat_prelim
                                   + z_train_1 / pi_z_hat_prelim * (d_train_1 - pi_d_z1_hat_prelim)
                                   - (1 - z_train_1) / (1 - pi_z_hat_prelim) * (d_train_1 - pi_d_z0_hat_prelim))

        def ipw_score(theta):
            sign = 2 * treatment - 1.0
            weights = sign * (z_train_1 / pi_z_hat_prelim - (1 - z_train_1) / (1 - pi_z_hat_prelim)) / comp_prob_prelim
            u = (d_train_1 == treatment) * (y_train_1 <= theta)
            v = -1. * quantile
            res = np.mean(weights * u + v)
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

        y_treat = y[d == treatment]
        coef_start_val = np.quantile(y_treat, q=quantile)
        coef_bounds = (y_treat.min(), y_treat.max())
        _, bracket_guess = get_bracket_guess(coef_start_val, coef_bounds)

        root_res = root_scalar(ipw_score,
                               bracket=bracket_guess,
                               method='brentq')
        ipw_est = root_res.root

        # use the preliminary estimates to fit the nuisance parameters on train_2
        d_train_2 = d[train_inds_2]
        y_train_2 = y[train_inds_2]
        x_train_2 = x[train_inds_2, :]
        z_train_2 = z[train_inds_2]

        # propensity for (D == treatment)*Ind(Y <= ipq_est) cond. on z == 0
        x_z0_train_2 = x_train_2[z_train_2 == 0, :]
        du_z0_train_2 = (d_train_2[z_train_2 == 0] == treatment) * (y_train_2[z_train_2 == 0] <= ipw_est)
        ml_pi_du_z0.fit(x_z0_train_2, du_z0_train_2)
        pi_du_z0_hat[test_inds] = ml_pi_du_z0.predict_proba(x[test_inds, :])[:, 1]

        # propensity for (D == treatment)*Ind(Y <= ipq_est) cond. on z == 1
        x_z1_train_2 = x_train_2[z_train_2 == 1, :]
        du_z1_train_2 = (d_train_2[z_train_2 == 1] == treatment) * (y_train_2[z_train_2 == 1] <= ipw_est)
        ml_pi_du_z1.fit(x_z1_train_2, du_z1_train_2)
        pi_du_z1_hat[test_inds] = ml_pi_du_z0.predict_proba(x[test_inds, :])[:, 1]

        # refit nuisance elements for the local potential quantile
        z_train = z[train_inds]
        x_train = x[train_inds]
        d_train = d[train_inds]

        # refit propensity for z (whole training set)
        ml_pi_z.fit(x_train, z_train)
        pi_z_hat[test_inds] = ml_pi_z.predict_proba(x[test_inds, :])[:, 1]

        # refit propensity for d == 1 cond. on z == 0 (whole training set)
        x_z0_train = x_train[z_train == 0, :]
        d_z0_train = d_train[z_train == 0]
        ml_pi_d_z0.fit(x_z0_train, d_z0_train)
        pi_d_z0_hat[test_inds] = ml_pi_d_z0.predict_proba(x[test_inds, :])[:, 1]

        # propensity for d == 1 cond. on z == 1 (whole training set)
        x_z1_train = x_train[z_train == 1, :]
        d_z1_train = d_train[z_train == 1]
        ml_pi_d_z1.fit(x_z1_train, d_z1_train)
        pi_d_z1_hat[test_inds] = ml_pi_d_z1.predict_proba(x[test_inds, :])[:, 1]

    # clip propensities
    pi_z_hat = _trimm(pi_z_hat, trimming_rule, trimming_threshold)
    pi_d_z0_hat = _trimm(pi_d_z0_hat, trimming_rule, trimming_threshold)
    pi_d_z1_hat = _trimm(pi_d_z1_hat, trimming_rule, trimming_threshold)
    pi_du_z0_hat = _trimm(pi_du_z0_hat, trimming_rule, trimming_threshold)
    pi_du_z1_hat = _trimm(pi_du_z1_hat, trimming_rule, trimming_threshold)

    # estimate final nuisance parameter
    comp_prob_hat = np.mean(pi_d_z1_hat - pi_d_z0_hat
                            + z / pi_z_hat * (d - pi_d_z1_hat)
                            - (1 - z) / (1 - pi_z_hat) * (d - pi_d_z0_hat))
    return pi_z_hat, pi_du_z0_hat, pi_du_z1_hat, comp_prob_hat, ipw_est


def lpq_dml1(y, d, z, pi_z, pi_du_z0, pi_du_z1, comp_prob, treatment, quantile, ipw_est, smpls):
    thetas = np.zeros(len(smpls))
    n_obs = len(y)
    for idx, (_, test_index) in enumerate(smpls):
        thetas[idx] = lpq_est(pi_z[test_index], pi_du_z0[test_index], pi_du_z1[test_index],
                              comp_prob, d[test_index], y[test_index], z[test_index], treatment, quantile, ipw_est)

    theta_hat = np.mean(thetas)

    se = np.sqrt(lpq_var_est(theta_hat, pi_z, pi_du_z0, pi_du_z1, comp_prob, d, y, z, treatment, quantile, n_obs))

    return theta_hat, se


def lpq_dml2(y, d, z, pi_z, pi_du_z0, pi_du_z1, comp_prob, treatment, quantile, ipw_est):
    n_obs = len(y)
    theta_hat = lpq_est(pi_z, pi_du_z0, pi_du_z1, comp_prob, d, y, z, treatment, quantile, ipw_est)

    se = np.sqrt(lpq_var_est(theta_hat, pi_z, pi_du_z0, pi_du_z1, comp_prob, d, y, z, treatment, quantile, n_obs))

    return theta_hat, se


def lpq_est(pi_z, pi_du_z0, pi_du_z1, comp_prob, d, y, z, treatment, quantile, ipw_est):
    def compute_score(coef):
        sign = 2 * treatment - 1.0
        score1 = pi_du_z1 - pi_du_z0
        score2 = (z / pi_z) * ((d == treatment) * (y <= coef) - pi_du_z1)
        score3 = (1 - z) / (1 - pi_z) * ((d == treatment) * (y <= coef) - pi_du_z0)
        score = sign * (score1 + score2 - score3) / comp_prob - quantile
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

    y_treat = y[d == treatment]
    coef_start_val = ipw_est
    coef_bounds = (y_treat.min(), y_treat.max())
    _, bracket_guess = get_bracket_guess(coef_start_val, coef_bounds)

    root_res = root_scalar(compute_score,
                           bracket=bracket_guess,
                           method='brentq')
    dml_est = root_res.root
    return dml_est


def lpq_var_est(coef, pi_z, pi_du_z0, pi_du_z1, comp_prob, d, y, z, treatment, quantile, n_obs, normalize=True, h=None):
    sign = 2 * treatment - 1.0
    score_weights = sign * ((z / pi_z) - (1 - z) / (1 - pi_z)) * (d == treatment) / comp_prob
    normalization = score_weights.mean()

    if normalize:
        score_weights /= normalization
    if h is None:
        h = np.power(n_obs, -0.2)
    u = (y - coef).reshape(-1, 1) / h
    kernel_est = np.exp(-1. * np.power(u, 2) / 2) / np.sqrt(2 * np.pi)

    deriv = np.multiply(score_weights, kernel_est.reshape(-1,)) / h
    J = np.mean(deriv)
    sign = 2 * treatment - 1.0
    score1 = pi_du_z1 - pi_du_z0
    score2 = (z / pi_z) * ((d == treatment) * (y <= coef) - pi_du_z1)
    score3 = (1 - z) / (1 - pi_z) * ((d == treatment) * (y <= coef) - pi_du_z0)
    score = sign * (score1 + score2 - score3) / comp_prob - quantile
    var_est = 1/n_obs * np.mean(np.square(score)) / np.square(J)
    return var_est
