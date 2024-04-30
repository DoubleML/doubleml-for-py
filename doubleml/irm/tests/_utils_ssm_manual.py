import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from ...tests._utils import fit_predict, fit_predict_proba, tune_grid_search
from ...utils._estimation import _predict_zero_one_propensity, _trimm


def fit_selection(y, x, d, z, s,
                  learner_g, learner_pi, learner_m,
                  all_smpls, score,
                  trimming_rule='truncate',
                  trimming_threshold=1e-2,
                  normalize_ipw=True,
                  n_rep=1,
                  g_d0_params=None, g_d1_params=None,
                  pi_params=None, m_params=None):
    n_obs = len(y)

    thetas = np.zeros(n_rep)
    ses = np.zeros(n_rep)

    all_g_d1_hat = list()
    all_g_d0_hat = list()
    all_pi_hat = list()
    all_m_hat = list()

    all_psi_a = list()
    all_psi_b = list()

    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]

        g_hat_d1_list, g_hat_d0_list, pi_hat_list, \
            m_hat_list = fit_nuisance_selection(y, x, d, z, s,
                                                learner_g, learner_pi, learner_m,
                                                smpls, score,
                                                trimming_rule=trimming_rule,
                                                trimming_threshold=trimming_threshold,
                                                g_d0_params=g_d0_params, g_d1_params=g_d1_params,
                                                pi_params=pi_params, m_params=m_params)
        all_g_d1_hat.append(g_hat_d1_list)
        all_g_d0_hat.append(g_hat_d0_list)
        all_pi_hat.append(pi_hat_list)
        all_m_hat.append(m_hat_list)

        g_hat_d1, g_hat_d0, pi_hat, m_hat = compute_selection(y, g_hat_d1_list, g_hat_d0_list, pi_hat_list, m_hat_list, smpls)

        dtreat = (d == 1)
        dcontrol = (d == 0)
        psi_a, psi_b = selection_score_elements(dtreat, dcontrol, g_hat_d1, g_hat_d0, pi_hat, m_hat,
                                                s, y, normalize_ipw)

        all_psi_a.append(psi_a)
        all_psi_b.append(psi_b)

        thetas[i_rep], ses[i_rep] = selection_dml2(psi_a, psi_b)

    theta = np.median(thetas)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(thetas - theta, 2)) / n_obs)

    res = {'theta': theta, 'se': se,
           'thetas': thetas, 'ses': ses,
           'all_g_d1_hat': all_g_d1_hat, 'all_g_d0_hat': all_g_d0_hat,
           'all_pi_hat': all_pi_hat, 'all_m_hat': all_m_hat,
           'all_psi_a': all_psi_a, 'all_psi_b': all_psi_b}

    return res


def fit_nuisance_selection(y, x, d, z, s,
                           learner_g, learner_pi, learner_m,
                           smpls, score,
                           trimming_rule='truncate',
                           trimming_threshold=1e-2,
                           g_d0_params=None, g_d1_params=None,
                           pi_params=None, m_params=None):

    ml_g_d1 = clone(learner_g)
    ml_g_d0 = clone(learner_g)
    ml_pi = clone(learner_pi)
    ml_m = clone(learner_m)

    if z is None:
        dx = np.column_stack((d, x))
    else:
        dx = np.column_stack((d, x, z))

    if score == 'missing-at-random':
        pi_hat_list = fit_predict_proba(s, dx, ml_pi, pi_params, smpls, trimming_threshold=trimming_threshold)

        m_hat_list = fit_predict_proba(d, x, ml_m, m_params, smpls)

        train_cond_d1_s1 = np.intersect1d(np.where(d == 1)[0], np.where(s == 1)[0])
        g_hat_d1_list = fit_predict(y, x, ml_g_d1, g_d1_params, smpls, train_cond=train_cond_d1_s1)

        train_cond_d0_s1 = np.intersect1d(np.where(d == 0)[0], np.where(s == 1)[0])
        g_hat_d0_list = fit_predict(y, x, ml_g_d0, g_d0_params, smpls, train_cond=train_cond_d0_s1)
    else:
        # initialize empty lists
        g_hat_d1_list = []
        g_hat_d0_list = []
        pi_hat_list = []
        m_hat_list = []

        # create strata for splitting
        strata = d.reshape(-1, 1) + 2 * s.reshape(-1, 1)

        # POTENTIAL OUTCOME Y(1)
        for i_fold, _ in enumerate(smpls):
            ml_g_d1 = clone(learner_g)
            ml_pi = clone(learner_pi)
            ml_m = clone(learner_m)

            # set the params for the nuisance learners
            if g_d1_params is not None:
                ml_g_d1.set_params(**g_d1_params[i_fold])
            if g_d0_params is not None:
                ml_g_d0.set_params(**g_d0_params[i_fold])
            if pi_params is not None:
                ml_pi.set_params(**pi_params[i_fold])
            if m_params is not None:
                ml_m.set_params(**m_params[i_fold])

            train_inds = smpls[i_fold][0]
            test_inds = smpls[i_fold][1]

            # start nested crossfitting
            train_inds_1, train_inds_2 = train_test_split(train_inds, test_size=0.5,
                                                          random_state=42, stratify=strata[train_inds])

            s_train_1 = s[train_inds_1]
            dx_train_1 = dx[train_inds_1, :]

            # preliminary propensity score for selection
            ml_pi_prelim = clone(ml_pi)
            # fit on first part of training set
            ml_pi_prelim.fit(dx_train_1, s_train_1)
            pi_hat_prelim = _predict_zero_one_propensity(ml_pi_prelim, dx)

            # predictions for small pi in denominator
            pi_hat = pi_hat_prelim[test_inds]

            # add selection indicator to covariates
            xpi = np.column_stack((x, pi_hat_prelim))

            # estimate propensity score p using the second training sample
            xpi_train_2 = xpi[train_inds_2, :]
            d_train_2 = d[train_inds_2]
            xpi_test = xpi[test_inds, :]

            ml_m.fit(xpi_train_2, d_train_2)

            m_hat = _predict_zero_one_propensity(ml_m, xpi_test)

            # estimate conditional outcome on second training sample -- treatment
            s1_d1_train_2_indices = np.intersect1d(np.where(d == 1)[0],
                                                   np.intersect1d(np.where(s == 1)[0], train_inds_2))
            xpi_s1_d1_train_2 = xpi[s1_d1_train_2_indices, :]
            y_s1_d1_train_2 = y[s1_d1_train_2_indices]

            ml_g_d1.fit(xpi_s1_d1_train_2, y_s1_d1_train_2)

            # predict conditional outcome
            g_hat_d1 = ml_g_d1.predict(xpi_test)

            # estimate conditional outcome on second training sample -- control
            s1_d0_train_2_indices = np.intersect1d(np.where(d == 0)[0],
                                                   np.intersect1d(np.where(s == 1)[0], train_inds_2))
            xpi_s1_d0_train_2 = xpi[s1_d0_train_2_indices, :]
            y_s1_d0_train_2 = y[s1_d0_train_2_indices]

            ml_g_d0.fit(xpi_s1_d0_train_2, y_s1_d0_train_2)

            # predict conditional outcome
            g_hat_d0 = ml_g_d0.predict(xpi_test)

            m_hat = _trimm(m_hat, trimming_rule, trimming_threshold)

            # append predictions on test sample to final list of predictions
            g_hat_d1_list.append(g_hat_d1)
            g_hat_d0_list.append(g_hat_d0)
            pi_hat_list.append(pi_hat)
            m_hat_list.append(m_hat)

    return g_hat_d1_list, g_hat_d0_list, pi_hat_list, m_hat_list


def compute_selection(y, g_hat_d1_list, g_hat_d0_list, pi_hat_list, m_hat_list, smpls):
    g_hat_d1 = np.full_like(y, np.nan, dtype='float64')
    g_hat_d0 = np.full_like(y, np.nan, dtype='float64')
    pi_hat = np.full_like(y, np.nan, dtype='float64')
    m_hat = np.full_like(y, np.nan, dtype='float64')

    for idx, (_, test_index) in enumerate(smpls):
        g_hat_d1[test_index] = g_hat_d1_list[idx]
        g_hat_d0[test_index] = g_hat_d0_list[idx]
        pi_hat[test_index] = pi_hat_list[idx]
        m_hat[test_index] = m_hat_list[idx]

    return g_hat_d1, g_hat_d0, pi_hat, m_hat


def selection_score_elements(dtreat, dcontrol, g_d1, g_d0,
                             pi, m, s, y, normalize_ipw):
    # psi_a
    psi_a = -1 * np.ones_like(y)

    # psi_b
    if normalize_ipw:
        weight_treat = sum(dtreat) / sum((dtreat * s) / (m * pi))
        weight_control = sum(dcontrol) / sum((dcontrol * s) / ((1 - m) * pi))

        psi_b1 = weight_treat * ((dtreat * s * (y - g_d1)) / (m * pi)) + g_d1
        psi_b0 = weight_control * ((dcontrol * s * (y - g_d0)) / ((1 - m) * pi)) + g_d0

    else:
        psi_b1 = (dtreat * s * (y - g_d1)) / (m * pi) + g_d1
        psi_b0 = (dcontrol * s * (y - g_d0)) / ((1 - m) * pi) + g_d0

    psi_b = psi_b1 - psi_b0

    return psi_a, psi_b


def selection_dml2(psi_a, psi_b):
    n_obs = len(psi_a)
    theta_hat = - np.mean(psi_b) / np.mean(psi_a)
    se = np.sqrt(var_selection(theta_hat, psi_a, psi_b, n_obs))

    return theta_hat, se


def var_selection(theta, psi_a, psi_b, n_obs):
    J = np.mean(psi_a)
    var = 1/n_obs * np.mean(np.power(np.multiply(psi_a, theta) + psi_b, 2)) / np.power(J, 2)
    return var


def tune_nuisance_ssm(y, x, d, z, s, ml_g, ml_pi, ml_m, smpls, score, n_folds_tune,
                      param_grid_g, param_grid_pi, param_grid_m):
    d0_s1 = np.intersect1d(np.where(d == 0)[0], np.where(s == 1)[0])
    d1_s1 = np.intersect1d(np.where(d == 1)[0], np.where(s == 1)[0])

    g0_tune_res = tune_grid_search(y, x, ml_g, smpls, param_grid_g, n_folds_tune,
                                   train_cond=d0_s1)
    g1_tune_res = tune_grid_search(y, x, ml_g, smpls, param_grid_g, n_folds_tune,
                                   train_cond=d1_s1)

    if score == 'nonignorable':
        dx = np.column_stack((x, d, z))
    else:
        dx = np.column_stack((x, d))

    pi_tune_res = tune_grid_search(s, dx, ml_pi, smpls, param_grid_pi, n_folds_tune)

    m_tune_res = tune_grid_search(d, x, ml_m, smpls, param_grid_m, n_folds_tune)

    g0_best_params = [xx.best_params_ for xx in g0_tune_res]
    g1_best_params = [xx.best_params_ for xx in g1_tune_res]
    pi_best_params = [xx.best_params_ for xx in pi_tune_res]
    m_best_params = [xx.best_params_ for xx in m_tune_res]

    return g0_best_params, g1_best_params, pi_best_params, m_best_params
