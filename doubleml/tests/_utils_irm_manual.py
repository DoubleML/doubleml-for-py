import numpy as np
from sklearn.base import clone

from ._utils_boot import boot_manual, draw_weights
from ._utils import fit_predict, fit_predict_proba, tune_grid_search


def fit_irm(y, x, d,
            learner_g, learner_m, all_smpls, dml_procedure, score,
            n_rep=1, g0_params=None, g1_params=None, m_params=None,
            trimming_threshold=1e-12):
    n_obs = len(y)

    thetas = np.zeros(n_rep)
    ses = np.zeros(n_rep)
    all_g_hat0 = list()
    all_g_hat1 = list()
    all_m_hat = list()
    all_p_hat = list()
    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]

        g_hat0, g_hat1, m_hat, p_hat = fit_nuisance_irm(y, x, d,
                                                        learner_g, learner_m, smpls,
                                                        score,
                                                        g0_params=g0_params, g1_params=g1_params, m_params=m_params,
                                                        trimming_threshold=trimming_threshold)

        all_g_hat0.append(g_hat0)
        all_g_hat1.append(g_hat1)
        all_m_hat.append(m_hat)
        all_p_hat.append(p_hat)

        if dml_procedure == 'dml1':
            thetas[i_rep], ses[i_rep] = irm_dml1(y, x, d,
                                                 g_hat0, g_hat1, m_hat, p_hat,
                                                 smpls, score)
        else:
            assert dml_procedure == 'dml2'
            thetas[i_rep], ses[i_rep] = irm_dml2(y, x, d,
                                                 g_hat0, g_hat1, m_hat, p_hat,
                                                 smpls, score)

    theta = np.median(thetas)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(thetas - theta, 2)) / n_obs)

    res = {'theta': theta, 'se': se,
           'thetas': thetas, 'ses': ses,
           'all_g_hat0': all_g_hat0, 'all_g_hat1': all_g_hat1, 'all_m_hat': all_m_hat, 'all_p_hat': all_p_hat}

    return res


def fit_nuisance_irm(y, x, d, learner_g, learner_m, smpls, score,
                     g0_params=None, g1_params=None, m_params=None,
                     trimming_threshold=1e-12):
    ml_g0 = clone(learner_g)
    ml_g1 = clone(learner_g)
    train_cond0 = np.where(d == 0)[0]
    g_hat0_list = fit_predict(y, x, ml_g0, g0_params, smpls,
                              train_cond=train_cond0)

    if score == 'ATE':
        train_cond1 = np.where(d == 1)[0]
        g_hat1_list = fit_predict(y, x, ml_g1, g1_params, smpls,
                                  train_cond=train_cond1)
    else:
        assert score == 'ATTE'
        g_hat1_list = list()
        for idx, _ in enumerate(smpls):
            # fill it up, but its not further used
            g_hat1_list.append(np.zeros_like(g_hat0_list[idx], dtype='float64'))

    ml_m = clone(learner_m)
    m_hat_list = fit_predict_proba(d, x, ml_m, m_params, smpls,
                                   trimming_threshold=trimming_threshold)

    p_hat_list = []
    for (_, test_index) in smpls:
        p_hat_list.append(np.mean(d[test_index]))

    return g_hat0_list, g_hat1_list, m_hat_list, p_hat_list


def tune_nuisance_irm(y, x, d, ml_g, ml_m, smpls, score, n_folds_tune,
                      param_grid_g, param_grid_m):
    train_cond0 = np.where(d == 0)[0]
    g0_tune_res = tune_grid_search(y, x, ml_g, smpls, param_grid_g, n_folds_tune,
                                   train_cond=train_cond0)

    if score == 'ATE':
        train_cond1 = np.where(d == 1)[0]
        g1_tune_res = tune_grid_search(y, x, ml_g, smpls, param_grid_g, n_folds_tune,
                                       train_cond=train_cond1)
        g1_best_params = [xx.best_params_ for xx in g1_tune_res]
    else:
        g1_best_params = None

    m_tune_res = tune_grid_search(d, x, ml_m, smpls, param_grid_m, n_folds_tune)

    g0_best_params = [xx.best_params_ for xx in g0_tune_res]
    m_best_params = [xx.best_params_ for xx in m_tune_res]

    return g0_best_params, g1_best_params, m_best_params


def compute_iivm_residuals(y, g_hat0_list, g_hat1_list, m_hat_list, p_hat_list, smpls):
    u_hat0 = np.full_like(y, np.nan, dtype='float64')
    u_hat1 = np.full_like(y, np.nan, dtype='float64')
    g_hat0 = np.full_like(y, np.nan, dtype='float64')
    g_hat1 = np.full_like(y, np.nan, dtype='float64')
    m_hat = np.full_like(y, np.nan, dtype='float64')
    p_hat = np.full_like(y, np.nan, dtype='float64')
    for idx, (_, test_index) in enumerate(smpls):
        u_hat0[test_index] = y[test_index] - g_hat0_list[idx]
        u_hat1[test_index] = y[test_index] - g_hat1_list[idx]
        g_hat0[test_index] = g_hat0_list[idx]
        g_hat1[test_index] = g_hat1_list[idx]
        m_hat[test_index] = m_hat_list[idx]
        p_hat[test_index] = p_hat_list[idx]

    return u_hat0, u_hat1, g_hat0, g_hat1, m_hat, p_hat


def irm_dml1(y, x, d, g_hat0_list, g_hat1_list, m_hat_list, p_hat_list, smpls, score):
    thetas = np.zeros(len(smpls))
    n_obs = len(y)
    u_hat0, u_hat1, g_hat0, g_hat1, m_hat, p_hat = compute_iivm_residuals(
        y, g_hat0_list, g_hat1_list, m_hat_list, p_hat_list, smpls)

    for idx, (_, test_index) in enumerate(smpls):
        thetas[idx] = irm_orth(g_hat0[test_index], g_hat1[test_index],
                               m_hat[test_index], p_hat[test_index],
                               u_hat0[test_index], u_hat1[test_index],
                               d[test_index], score)
    theta_hat = np.mean(thetas)

    if len(smpls) > 1:
        se = np.sqrt(var_irm(theta_hat, g_hat0, g_hat1,
                             m_hat, p_hat,
                             u_hat0, u_hat1,
                             d, score, n_obs))
    else:
        assert len(smpls) == 1
        test_index = smpls[0][1]
        n_obs = len(test_index)
        se = np.sqrt(var_irm(theta_hat, g_hat0[test_index], g_hat1[test_index],
                             m_hat[test_index], p_hat[test_index],
                             u_hat0[test_index], u_hat1[test_index],
                             d[test_index], score, n_obs))

    return theta_hat, se


def irm_dml2(y, x, d, g_hat0_list, g_hat1_list, m_hat_list, p_hat_list, smpls, score):
    n_obs = len(y)
    u_hat0, u_hat1, g_hat0, g_hat1, m_hat, p_hat = compute_iivm_residuals(
        y, g_hat0_list, g_hat1_list, m_hat_list, p_hat_list, smpls)
    theta_hat = irm_orth(g_hat0, g_hat1, m_hat, p_hat,
                         u_hat0, u_hat1, d, score)
    se = np.sqrt(var_irm(theta_hat, g_hat0, g_hat1,
                         m_hat, p_hat,
                         u_hat0, u_hat1,
                         d, score, n_obs))

    return theta_hat, se


def var_irm(theta, g_hat0, g_hat1, m_hat, p_hat, u_hat0, u_hat1, d, score, n_obs):
    if score == 'ATE':
        var = 1/n_obs * np.mean(np.power(g_hat1 - g_hat0
                                         + np.divide(np.multiply(d, u_hat1), m_hat)
                                         - np.divide(np.multiply(1.-d, u_hat0), 1.-m_hat) - theta, 2))
    else:
        assert score == 'ATTE'
        var = 1/n_obs * np.mean(np.power(np.divide(np.multiply(d, u_hat0), p_hat)
                                         - np.divide(np.multiply(m_hat, np.multiply(1.-d, u_hat0)),
                                                     np.multiply(p_hat, (1.-m_hat)))
                                         - theta * np.divide(d, p_hat), 2)) \
            / np.power(np.mean(np.divide(d, p_hat)), 2)

    return var


def irm_orth(g_hat0, g_hat1, m_hat, p_hat, u_hat0, u_hat1, d, score):
    if score == 'ATE':
        res = np.mean(g_hat1 - g_hat0
                      + np.divide(np.multiply(d, u_hat1), m_hat)
                      - np.divide(np.multiply(1.-d, u_hat0), 1.-m_hat))
    else:
        assert score == 'ATTE'
        res = np.mean(np.divide(np.multiply(d, u_hat0), p_hat)
                      - np.divide(np.multiply(m_hat, np.multiply(1.-d, u_hat0)),
                                  np.multiply(p_hat, (1.-m_hat)))) \
            / np.mean(np.divide(d, p_hat))

    return res


def boot_irm(y, d, thetas, ses, all_g_hat0, all_g_hat1, all_m_hat, all_p_hat,
             all_smpls, score, bootstrap, n_rep_boot,
             n_rep=1, apply_cross_fitting=True):
    all_boot_theta = list()
    all_boot_t_stat = list()
    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]
        if apply_cross_fitting:
            n_obs = len(y)
        else:
            test_index = smpls[0][1]
            n_obs = len(test_index)
        weights = draw_weights(bootstrap, n_rep_boot, n_obs)
        boot_theta, boot_t_stat = boot_irm_single_split(
            thetas[i_rep], y, d,
            all_g_hat0[i_rep], all_g_hat1[i_rep], all_m_hat[i_rep], all_p_hat[i_rep], smpls,
            score, ses[i_rep], weights, n_rep_boot, apply_cross_fitting)
        all_boot_theta.append(boot_theta)
        all_boot_t_stat.append(boot_t_stat)

    boot_theta = np.hstack(all_boot_theta)
    boot_t_stat = np.hstack(all_boot_t_stat)

    return boot_theta, boot_t_stat


def boot_irm_single_split(theta, y, d, g_hat0_list, g_hat1_list, m_hat_list, p_hat_list,
                          smpls, score, se, weights, n_rep_boot, apply_cross_fitting):
    u_hat0, u_hat1, g_hat0, g_hat1, m_hat, p_hat = compute_iivm_residuals(
        y, g_hat0_list, g_hat1_list, m_hat_list, p_hat_list, smpls)

    if apply_cross_fitting:
        if score == 'ATE':
            J = -1.0
        else:
            assert score == 'ATTE'
            J = np.mean(-np.divide(d, p_hat))
    else:
        test_index = smpls[0][1]
        if score == 'ATE':
            J = -1.0
        else:
            assert score == 'ATTE'
            J = np.mean(-np.divide(d[test_index], p_hat[test_index]))

    if score == 'ATE':
        psi = g_hat1 - g_hat0 \
                + np.divide(np.multiply(d, u_hat1), m_hat) \
                - np.divide(np.multiply(1.-d, u_hat0), 1.-m_hat) - theta
    else:
        assert score == 'ATTE'
        psi = np.divide(np.multiply(d, u_hat0), p_hat) \
            - np.divide(np.multiply(m_hat, np.multiply(1.-d, u_hat0)),
                        np.multiply(p_hat, (1.-m_hat))) \
            - theta * np.divide(d, p_hat)

    boot_theta, boot_t_stat = boot_manual(psi, J, smpls, se, weights, n_rep_boot, apply_cross_fitting)

    return boot_theta, boot_t_stat
