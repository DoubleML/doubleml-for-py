import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.base import clone

from ._utils_boot import boot_manual, draw_weights


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
    g_hat0_list = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if g0_params is not None:
            ml_g0.set_params(**g0_params[idx])
        train_index0 = np.intersect1d(np.where(z == 0)[0], train_index)
        g_hat0_list.append(ml_g0.fit(x[train_index0], y[train_index0]).predict(x[test_index]))

    ml_g1 = clone(learner_g)
    g_hat1_list = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if g1_params is not None:
            ml_g1.set_params(**g1_params[idx])
        train_index1 = np.intersect1d(np.where(z == 1)[0], train_index)
        g_hat1_list.append(ml_g1.fit(x[train_index1], y[train_index1]).predict(x[test_index]))

    ml_m = clone(learner_m)
    m_hat_list = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if m_params is not None:
            ml_m.set_params(**m_params[idx])
        xx = ml_m.fit(x[train_index], z[train_index]).predict_proba(x[test_index])[:, 1]
        if trimming_threshold > 0:
            xx[xx < trimming_threshold] = trimming_threshold
            xx[xx > 1 - trimming_threshold] = 1 - trimming_threshold
        m_hat_list.append(xx)

    ml_r0 = clone(learner_r)
    r_hat0_list = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if r0_params is not None:
            ml_r0.set_params(**r0_params[idx])
        train_index0 = np.intersect1d(np.where(z == 0)[0], train_index)
        if always_takers:
            r_hat0_list.append(ml_r0.fit(x[train_index0], d[train_index0]).predict_proba(x[test_index])[:, 1])
        else:
            r_hat0_list.append(np.zeros_like(d[test_index]))

    ml_r1 = clone(learner_r)
    r_hat1_list = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if r1_params is not None:
            ml_r1.set_params(**r1_params[idx])
        train_index1 = np.intersect1d(np.where(z == 1)[0], train_index)
        if never_takers:
            r_hat1_list.append(ml_r1.fit(x[train_index1], d[train_index1]).predict_proba(x[test_index])[:, 1])
        else:
            r_hat1_list.append(np.ones_like(d[test_index]))

    return g_hat0_list, g_hat1_list, m_hat_list, r_hat0_list, r_hat1_list


def tune_nuisance_iivm(y, x, d, z, ml_g, ml_m, ml_r, smpls, n_folds_tune,
                       param_grid_g, param_grid_m, param_grid_r,
                       always_takers=True, never_takers=True):
    g0_tune_res = [None] * len(smpls)
    for idx, (train_index, _) in enumerate(smpls):
        g0_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
        g0_grid_search = GridSearchCV(ml_g, param_grid_g,
                                      cv=g0_tune_resampling)
        train_index0 = np.intersect1d(np.where(z == 0)[0], train_index)
        g0_tune_res[idx] = g0_grid_search.fit(x[train_index0, :], y[train_index0])

    g1_tune_res = [None] * len(smpls)
    for idx, (train_index, _) in enumerate(smpls):
        g1_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
        g1_grid_search = GridSearchCV(ml_g, param_grid_g,
                                      cv=g1_tune_resampling)
        train_index1 = np.intersect1d(np.where(z == 1)[0], train_index)
        g1_tune_res[idx] = g1_grid_search.fit(x[train_index1, :], y[train_index1])

    m_tune_res = [None] * len(smpls)
    for idx, (train_index, _) in enumerate(smpls):
        m_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
        m_grid_search = GridSearchCV(ml_m, param_grid_m,
                                     cv=m_tune_resampling)
        m_tune_res[idx] = m_grid_search.fit(x[train_index, :], z[train_index])

    if always_takers:
        r0_tune_res = [None] * len(smpls)
        for idx, (train_index, _) in enumerate(smpls):
            r0_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
            r0_grid_search = GridSearchCV(ml_r, param_grid_r,
                                          cv=r0_tune_resampling)
            train_index0 = np.intersect1d(np.where(z == 0)[0], train_index)
            r0_tune_res[idx] = r0_grid_search.fit(x[train_index0, :], d[train_index0])
        r0_best_params = [xx.best_params_ for xx in r0_tune_res]
    else:
        r0_best_params = None

    if never_takers:
        r1_tune_res = [None] * len(smpls)
        for idx, (train_index, _) in enumerate(smpls):
            r1_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
            r1_grid_search = GridSearchCV(ml_r, param_grid_r,
                                          cv=r1_tune_resampling)
            train_index1 = np.intersect1d(np.where(z == 1)[0], train_index)
            r1_tune_res[idx] = r1_grid_search.fit(x[train_index1, :], d[train_index1])
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
        boot_theta, boot_t_stat = boot_iivm_single_split(
            thetas[i_rep], y, d, z,
            all_g_hat0[i_rep], all_g_hat1[i_rep], all_m_hat[i_rep], all_r_hat0[i_rep], all_r_hat1[i_rep],
            all_smpls[i_rep], score, ses[i_rep], bootstrap, n_rep_boot, apply_cross_fitting)
        all_boot_theta.append(boot_theta)
        all_boot_t_stat.append(boot_t_stat)

    boot_theta = np.hstack(all_boot_theta)
    boot_t_stat = np.hstack(all_boot_t_stat)

    return boot_theta, boot_t_stat


def boot_iivm_single_split(theta, y, d, z, g_hat0, g_hat1, m_hat, r_hat0, r_hat1,
                           smpls, score, se, bootstrap, n_rep, apply_cross_fitting=True):
    if apply_cross_fitting:
        n_obs = len(y)
    else:
        test_index = smpls[0][1]
        n_obs = len(test_index)
    weights = draw_weights(bootstrap, n_rep, n_obs)
    assert np.isscalar(theta)
    boot_theta, boot_t_stat = boot_iivm_single_treat(theta, y, d, z, g_hat0, g_hat1, m_hat, r_hat0, r_hat1,
                                                     smpls, score, se, weights, n_rep, apply_cross_fitting)
    return boot_theta, boot_t_stat


def boot_iivm_single_treat(theta, y, d, z, g_hat0_list, g_hat1_list, m_hat_list, r_hat0_list, r_hat1_list,
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
