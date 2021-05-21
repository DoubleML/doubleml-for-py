import numpy as np
from sklearn.base import clone

from ._utils_boot import boot_manual, draw_weights
from ._utils import fit_predict, fit_predict_proba, tune_grid_search


def fit_iivm(y, x, d, z,
             learner_g, learner_m, learner_r, all_smpls, dml_procedure, score,
             n_rep=1, g0_params=None, g1_params=None, m_params=None, r0_params=None, r1_params=None,
             trimming_threshold=1e-12, always_takers=True, never_takers=True):
    n_obs = len(y)

    thetas = np.zeros(n_rep)
    ses = np.zeros(n_rep)
    all_g_hat0 = list()
    all_g_hat1 = list()
    all_m_hat = list()
    all_r_hat0 = list()
    all_r_hat1 = list()
    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]

        g_hat0, g_hat1, m_hat, r_hat0, r_hat1 = fit_nuisance_iivm(
            y, x, d, z,
            learner_g, learner_m, learner_r, smpls,
            g0_params=g0_params, g1_params=g1_params, m_params=m_params, r0_params=r0_params, r1_params=r1_params,
            trimming_threshold=trimming_threshold, always_takers=always_takers, never_takers=never_takers)

        all_g_hat0.append(g_hat0)
        all_g_hat1.append(g_hat1)
        all_m_hat.append(m_hat)
        all_r_hat0.append(r_hat0)
        all_r_hat1.append(r_hat1)

        if dml_procedure == 'dml1':
            thetas[i_rep], ses[i_rep] = iivm_dml1(y, x, d, z,
                                                  g_hat0, g_hat1, m_hat, r_hat0, r_hat1,
                                                  smpls, score)
        else:
            assert dml_procedure == 'dml2'
            thetas[i_rep], ses[i_rep] = iivm_dml2(y, x, d, z,
                                                  g_hat0, g_hat1, m_hat, r_hat0, r_hat1,
                                                  smpls, score)

    theta = np.median(thetas)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(thetas - theta, 2)) / n_obs)

    res = {'theta': theta, 'se': se,
           'thetas': thetas, 'ses': ses,
           'all_g_hat0': all_g_hat0, 'all_g_hat1': all_g_hat1,
           'all_m_hat': all_m_hat, 'all_r_hat0': all_r_hat0, 'all_r_hat1': all_r_hat1}

    return res


def fit_nuisance_iivm(y, x, d, z, learner_g, learner_m, learner_r, smpls,
                      g0_params=None, g1_params=None, m_params=None, r0_params=None, r1_params=None,
                      trimming_threshold=1e-12, always_takers=True, never_takers=True):
    ml_g0 = clone(learner_g)
    train_cond0 = np.where(z == 0)[0]
    g_hat0_list = fit_predict(y, x, ml_g0, g0_params, smpls,
                              train_cond=train_cond0)

    ml_g1 = clone(learner_g)
    train_cond1 = np.where(z == 1)[0]
    g_hat1_list = fit_predict(y, x, ml_g1, g1_params, smpls,
                              train_cond=train_cond1)

    ml_m = clone(learner_m)
    m_hat_list = fit_predict_proba(z, x, ml_m, m_params, smpls,
                                   trimming_threshold=trimming_threshold)

    ml_r0 = clone(learner_r)
    if always_takers:
        r_hat0_list = fit_predict_proba(d, x, ml_r0, r0_params, smpls,
                                        train_cond=train_cond0)
    else:
        r_hat0_list = []
        for (_, test_index) in smpls:
            r_hat0_list.append(np.zeros_like(d[test_index]))

    ml_r1 = clone(learner_r)
    if never_takers:
        r_hat1_list = fit_predict_proba(d, x, ml_r1, r1_params, smpls,
                                        train_cond=train_cond1)
    else:
        r_hat1_list = []
        for (_, test_index) in smpls:
            r_hat1_list.append(np.ones_like(d[test_index]))

    return g_hat0_list, g_hat1_list, m_hat_list, r_hat0_list, r_hat1_list


def tune_nuisance_iivm(y, x, d, z, ml_g, ml_m, ml_r, smpls, n_folds_tune,
                       param_grid_g, param_grid_m, param_grid_r,
                       always_takers=True, never_takers=True):
    train_cond0 = np.where(z == 0)[0]
    g0_tune_res = tune_grid_search(y, x, ml_g, smpls, param_grid_g, n_folds_tune,
                                   train_cond=train_cond0)

    train_cond1 = np.where(z == 1)[0]
    g1_tune_res = tune_grid_search(y, x, ml_g, smpls, param_grid_g, n_folds_tune,
                                   train_cond=train_cond1)

    m_tune_res = tune_grid_search(z, x, ml_m, smpls, param_grid_m, n_folds_tune)

    if always_takers:
        r0_tune_res = tune_grid_search(d, x, ml_r, smpls, param_grid_r, n_folds_tune,
                                       train_cond=train_cond0)
        r0_best_params = [xx.best_params_ for xx in r0_tune_res]
    else:
        r0_best_params = None

    if never_takers:
        r1_tune_res = tune_grid_search(d, x, ml_r, smpls, param_grid_r, n_folds_tune,
                                       train_cond=train_cond1)
        r1_best_params = [xx.best_params_ for xx in r1_tune_res]
    else:
        r1_best_params = None

    g0_best_params = [xx.best_params_ for xx in g0_tune_res]
    g1_best_params = [xx.best_params_ for xx in g1_tune_res]
    m_best_params = [xx.best_params_ for xx in m_tune_res]

    return g0_best_params, g1_best_params, m_best_params, r0_best_params, r1_best_params


def compute_iivm_residuals(y, d, g_hat0_list, g_hat1_list, m_hat_list, r_hat0_list, r_hat1_list, smpls):
    u_hat0 = np.full_like(y, np.nan, dtype='float64')
    u_hat1 = np.full_like(y, np.nan, dtype='float64')
    w_hat0 = np.full_like(y, np.nan, dtype='float64')
    w_hat1 = np.full_like(y, np.nan, dtype='float64')
    g_hat0 = np.full_like(y, np.nan, dtype='float64')
    g_hat1 = np.full_like(y, np.nan, dtype='float64')
    r_hat0 = np.full_like(y, np.nan, dtype='float64')
    r_hat1 = np.full_like(y, np.nan, dtype='float64')
    m_hat = np.full_like(y, np.nan, dtype='float64')
    for idx, (_, test_index) in enumerate(smpls):
        u_hat0[test_index] = y[test_index] - g_hat0_list[idx]
        u_hat1[test_index] = y[test_index] - g_hat1_list[idx]
        w_hat0[test_index] = d[test_index] - r_hat0_list[idx]
        w_hat1[test_index] = d[test_index] - r_hat1_list[idx]
        g_hat0[test_index] = g_hat0_list[idx]
        g_hat1[test_index] = g_hat1_list[idx]
        m_hat[test_index] = m_hat_list[idx]
        r_hat0[test_index] = r_hat0_list[idx]
        r_hat1[test_index] = r_hat1_list[idx]

    return u_hat0, u_hat1, w_hat0, w_hat1, g_hat0, g_hat1, m_hat, r_hat0, r_hat1


def iivm_dml1(y, x, d, z, g_hat0_list, g_hat1_list, m_hat_list, r_hat0_list, r_hat1_list, smpls, score):
    thetas = np.zeros(len(smpls))
    n_obs = len(y)
    u_hat0, u_hat1, w_hat0, w_hat1, g_hat0, g_hat1, m_hat, r_hat0, r_hat1 = compute_iivm_residuals(
        y, d, g_hat0_list, g_hat1_list, m_hat_list, r_hat0_list, r_hat1_list, smpls)

    for idx, (_, test_index) in enumerate(smpls):
        thetas[idx] = iivm_orth(g_hat0[test_index], g_hat1[test_index],
                                m_hat[test_index],
                                r_hat0[test_index], r_hat1[test_index],
                                u_hat0[test_index], u_hat1[test_index],
                                w_hat0[test_index], w_hat1[test_index],
                                z[test_index], score)
    theta_hat = np.mean(thetas)

    if len(smpls) > 1:
        se = np.sqrt(var_iivm(theta_hat, g_hat0, g_hat1,
                              m_hat, r_hat0, r_hat1,
                              u_hat0, u_hat1, w_hat0, w_hat1,
                              z, score, n_obs))
    else:
        assert len(smpls) == 1
        test_index = smpls[0][1]
        n_obs = len(test_index)
        se = np.sqrt(var_iivm(theta_hat, g_hat0[test_index], g_hat1[test_index],
                              m_hat[test_index], r_hat0[test_index], r_hat1[test_index],
                              u_hat0[test_index], u_hat1[test_index], w_hat0[test_index], w_hat1[test_index],
                              z[test_index], score, n_obs))

    return theta_hat, se


def iivm_dml2(y, x, d, z, g_hat0_list, g_hat1_list, m_hat_list, r_hat0_list, r_hat1_list, smpls, score):
    n_obs = len(y)
    u_hat0, u_hat1, w_hat0, w_hat1, g_hat0, g_hat1, m_hat, r_hat0, r_hat1 = compute_iivm_residuals(
        y, d, g_hat0_list, g_hat1_list, m_hat_list, r_hat0_list, r_hat1_list, smpls)
    theta_hat = iivm_orth(g_hat0, g_hat1, m_hat, r_hat0, r_hat1,
                          u_hat0, u_hat1, w_hat0, w_hat1, z, score)
    se = np.sqrt(var_iivm(theta_hat, g_hat0, g_hat1,
                          m_hat, r_hat0, r_hat1,
                          u_hat0, u_hat1, w_hat0, w_hat1,
                          z, score, n_obs))

    return theta_hat, se


def var_iivm(theta, g_hat0, g_hat1, m_hat, r_hat0, r_hat1, u_hat0, u_hat1, w_hat0, w_hat1, z, score, n_obs):
    assert score == 'LATE'
    var = 1/n_obs * np.mean(np.power(g_hat1 - g_hat0
                                     + np.divide(np.multiply(z, u_hat1), m_hat)
                                     - np.divide(np.multiply(1.-z, u_hat0), 1.-m_hat)
                                     - theta*(r_hat1 - r_hat0
                                              + np.divide(np.multiply(z, w_hat1), m_hat)
                                              - np.divide(np.multiply(1.-z, w_hat0), 1.-m_hat)), 2)) \
        / np.power(np.mean(r_hat1 - r_hat0
                   + np.divide(np.multiply(z, w_hat1), m_hat)
                   - np.divide(np.multiply(1.-z, w_hat0), 1.-m_hat)), 2)

    return var


def iivm_orth(g_hat0, g_hat1, m_hat, r_hat0, r_hat1, u_hat0, u_hat1, w_hat0, w_hat1, z, score):
    assert score == 'LATE'
    res = np.mean(g_hat1 - g_hat0
                  + np.divide(np.multiply(z, u_hat1), m_hat)
                  - np.divide(np.multiply(1.-z, u_hat0), 1.-m_hat)) \
        / np.mean(r_hat1 - r_hat0
                  + np.divide(np.multiply(z, w_hat1), m_hat)
                  - np.divide(np.multiply(1.-z, w_hat0), 1.-m_hat))

    return res


def boot_iivm(y, d, z, thetas, ses, all_g_hat0, all_g_hat1, all_m_hat, all_r_hat0, all_r_hat1,
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
        boot_theta, boot_t_stat = boot_iivm_single_split(
            thetas[i_rep], y, d, z,
            all_g_hat0[i_rep], all_g_hat1[i_rep], all_m_hat[i_rep], all_r_hat0[i_rep], all_r_hat1[i_rep],
            smpls, score, ses[i_rep], weights, n_rep_boot, apply_cross_fitting)
        all_boot_theta.append(boot_theta)
        all_boot_t_stat.append(boot_t_stat)

    boot_theta = np.hstack(all_boot_theta)
    boot_t_stat = np.hstack(all_boot_t_stat)

    return boot_theta, boot_t_stat


def boot_iivm_single_split(theta, y, d, z, g_hat0_list, g_hat1_list, m_hat_list, r_hat0_list, r_hat1_list,
                           smpls, score, se, weights, n_rep, apply_cross_fitting):
    assert score == 'LATE'
    u_hat0, u_hat1, w_hat0, w_hat1, g_hat0, g_hat1, m_hat, r_hat0, r_hat1 = compute_iivm_residuals(
        y, d, g_hat0_list, g_hat1_list, m_hat_list, r_hat0_list, r_hat1_list, smpls)

    if apply_cross_fitting:
        J = np.mean(-(r_hat1 - r_hat0
                      + np.divide(np.multiply(z, w_hat1), m_hat)
                      - np.divide(np.multiply(1. - z, w_hat0), 1. - m_hat)))
    else:
        test_index = smpls[0][1]
        J = np.mean(-(r_hat1[test_index] - r_hat0[test_index]
                      + np.divide(np.multiply(z[test_index], w_hat1[test_index]), m_hat[test_index])
                      - np.divide(np.multiply(1. - z[test_index], w_hat0[test_index]),
                                  1. - m_hat[test_index])))

    psi = g_hat1 - g_hat0 \
        + np.divide(np.multiply(z, u_hat1), m_hat) \
        - np.divide(np.multiply(1.-z, u_hat0), 1.-m_hat) \
        - theta*(r_hat1 - r_hat0
                 + np.divide(np.multiply(z, w_hat1), m_hat)
                 - np.divide(np.multiply(1.-z, w_hat0), 1.-m_hat))

    boot_theta, boot_t_stat = boot_manual(psi, J, smpls, se, weights, n_rep, apply_cross_fitting)

    return boot_theta, boot_t_stat
