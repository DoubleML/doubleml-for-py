import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.base import clone

from ._utils_boot import boot_manual, draw_weights


def fit_nuisance_irm(y, x, d, learner_m, learner_g, smpls, score,
                     g0_params=None, g1_params=None, m_params=None,
                     trimming_threshold=1e-12):
    ml_g0 = clone(learner_g)
    ml_g1 = clone(learner_g)
    g_hat0 = []
    g_hat1 = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if g0_params is not None:
            ml_g0.set_params(**g0_params[idx])
        train_index0 = np.intersect1d(np.where(d == 0)[0], train_index)
        g_hat0.append(ml_g0.fit(x[train_index0], y[train_index0]).predict(x[test_index]))

    if score == 'ATE':
        for idx, (train_index, test_index) in enumerate(smpls):
            if g1_params is not None:
                ml_g1.set_params(**g1_params[idx])
            train_index1 = np.intersect1d(np.where(d == 1)[0], train_index)
            g_hat1.append(ml_g1.fit(x[train_index1], y[train_index1]).predict(x[test_index]))
    else:
        assert score == 'ATTE'
        for idx, _ in enumerate(smpls):
            # fill it up, but its not further used
            g_hat1.append(np.zeros_like(g_hat0[idx], dtype='float64'))

    ml_m = clone(learner_m)
    m_hat = []
    p_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if m_params is not None:
            ml_m.set_params(**m_params[idx])
        p_hat.append(np.mean(d[test_index]))
        xx = ml_m.fit(x[train_index], d[train_index]).predict_proba(x[test_index])[:, 1]
        if trimming_threshold > 0:
            xx[xx < trimming_threshold] = trimming_threshold
            xx[xx > 1 - trimming_threshold] = 1 - trimming_threshold
        m_hat.append(xx)

    return g_hat0, g_hat1, m_hat, p_hat


def tune_nuisance_irm(y, x, d, ml_m, ml_g, smpls, score, n_folds_tune,
                      param_grid_g, param_grid_m):
    g0_tune_res = [None] * len(smpls)
    for idx, (train_index, _) in enumerate(smpls):
        g0_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
        g0_grid_search = GridSearchCV(ml_g, param_grid_g,
                                      cv=g0_tune_resampling)
        train_index0 = np.intersect1d(np.where(d == 0)[0], train_index)
        g0_tune_res[idx] = g0_grid_search.fit(x[train_index0, :], y[train_index0])

    if score == 'ATE':
        g1_tune_res = [None] * len(smpls)
        for idx, (train_index, _) in enumerate(smpls):
            g1_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
            g1_grid_search = GridSearchCV(ml_g, param_grid_g,
                                          cv=g1_tune_resampling)
            train_index1 = np.intersect1d(np.where(d == 1)[0], train_index)
            g1_tune_res[idx] = g1_grid_search.fit(x[train_index1, :], y[train_index1])

    m_tune_res = [None] * len(smpls)
    for idx, (train_index, _) in enumerate(smpls):
        m_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
        m_grid_search = GridSearchCV(ml_m, param_grid_m,
                                     cv=m_tune_resampling)
        m_tune_res[idx] = m_grid_search.fit(x[train_index, :], d[train_index])

    g0_best_params = [xx.best_params_ for xx in g0_tune_res]
    if score == 'ATTE':
        g1_best_params = None
    else:
        assert score == 'ATE'
        g1_best_params = [xx.best_params_ for xx in g1_tune_res]
    m_best_params = [xx.best_params_ for xx in m_tune_res]

    return g0_best_params, g1_best_params, m_best_params


def irm_dml1(y, x, d, g_hat0, g_hat1, m_hat, p_hat, smpls, score):
    thetas = np.zeros(len(smpls))
    n_obs = len(y)

    for idx, (_, test_index) in enumerate(smpls):
        u_hat0 = y[test_index] - g_hat0[idx]
        u_hat1 = y[test_index] - g_hat1[idx]
        thetas[idx] = irm_orth(g_hat0[idx], g_hat1[idx],
                               m_hat[idx], p_hat[idx],
                               u_hat0, u_hat1,
                               d[test_index], score)
    theta_hat = np.mean(thetas)

    u_hat0 = np.zeros_like(y, dtype='float64')
    u_hat1 = np.zeros_like(y, dtype='float64')
    g_hat0_all = np.zeros_like(y, dtype='float64')
    g_hat1_all = np.zeros_like(y, dtype='float64')
    m_hat_all = np.zeros_like(y, dtype='float64')
    p_hat_all = np.zeros_like(y, dtype='float64')
    for idx, (_, test_index) in enumerate(smpls):
        u_hat0[test_index] = y[test_index] - g_hat0[idx]
        u_hat1[test_index] = y[test_index] - g_hat1[idx]
        g_hat0_all[test_index] = g_hat0[idx]
        g_hat1_all[test_index] = g_hat1[idx]
        m_hat_all[test_index] = m_hat[idx]
        p_hat_all[test_index] = p_hat[idx]
    se = np.sqrt(var_irm(theta_hat, g_hat0_all, g_hat1_all,
                         m_hat_all, p_hat_all,
                         u_hat0, u_hat1,
                         d, score, n_obs))

    return theta_hat, se


def irm_dml2(y, x, d, g_hat0, g_hat1, m_hat, p_hat, smpls, score):
    n_obs = len(y)
    u_hat0 = np.zeros_like(y, dtype='float64')
    u_hat1 = np.zeros_like(y, dtype='float64')
    g_hat0_all = np.zeros_like(y, dtype='float64')
    g_hat1_all = np.zeros_like(y, dtype='float64')
    m_hat_all = np.zeros_like(y, dtype='float64')
    p_hat_all = np.zeros_like(y, dtype='float64')
    for idx, (_, test_index) in enumerate(smpls):
        u_hat0[test_index] = y[test_index] - g_hat0[idx]
        u_hat1[test_index] = y[test_index] - g_hat1[idx]
        g_hat0_all[test_index] = g_hat0[idx]
        g_hat1_all[test_index] = g_hat1[idx]
        m_hat_all[test_index] = m_hat[idx]
        p_hat_all[test_index] = p_hat[idx]
    theta_hat = irm_orth(g_hat0_all, g_hat1_all, m_hat_all, p_hat_all,
                         u_hat0, u_hat1, d, score)
    se = np.sqrt(var_irm(theta_hat, g_hat0_all, g_hat1_all,
                         m_hat_all, p_hat_all,
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


def boot_irm(theta, y, d, g_hat0, g_hat1, m_hat, p_hat, smpls, score, se, bootstrap, n_rep, dml_procedure):
    n_obs = len(y)
    weights = draw_weights(bootstrap, n_rep, n_obs)
    assert np.isscalar(theta)
    boot_theta, boot_t_stat = boot_irm_single_treat(theta, y, d, g_hat0, g_hat1, m_hat, p_hat,
                                                    smpls, score, se, weights, n_rep, dml_procedure)
    return boot_theta, boot_t_stat


def boot_irm_single_treat(theta, y, d, g_hat0, g_hat1, m_hat, p_hat, smpls, score, se, weights, n_rep, dml_procedure):
    u_hat0 = np.zeros_like(y, dtype='float64')
    u_hat1 = np.zeros_like(y, dtype='float64')
    g_hat0_all = np.zeros_like(y, dtype='float64')
    g_hat1_all = np.zeros_like(y, dtype='float64')
    m_hat_all = np.zeros_like(y, dtype='float64')
    p_hat_all = np.zeros_like(y, dtype='float64')
    n_folds = len(smpls)
    J = np.zeros(n_folds)
    for idx, (_, test_index) in enumerate(smpls):
        u_hat0[test_index] = y[test_index] - g_hat0[idx]
        u_hat1[test_index] = y[test_index] - g_hat1[idx]
        g_hat0_all[test_index] = g_hat0[idx]
        g_hat1_all[test_index] = g_hat1[idx]
        m_hat_all[test_index] = m_hat[idx]
        p_hat_all[test_index] = p_hat[idx]
        if dml_procedure == 'dml1':
            if score == 'ATE':
                J[idx] = -1.0
            else:
                assert score == 'ATTE'
                J[idx] = np.mean(-np.divide(d[test_index], p_hat_all[test_index]))

    if dml_procedure == 'dml2':
        if score == 'ATE':
            J = -1.0
        else:
            assert score == 'ATTE'
            J = np.mean(-np.divide(d, p_hat_all))

    if score == 'ATE':
        psi = g_hat1_all - g_hat0_all \
                + np.divide(np.multiply(d, u_hat1), m_hat_all) \
                - np.divide(np.multiply(1.-d, u_hat0), 1.-m_hat_all) - theta
    else:
        assert score == 'ATTE'
        psi = np.divide(np.multiply(d, u_hat0), p_hat_all) \
            - np.divide(np.multiply(m_hat_all, np.multiply(1.-d, u_hat0)),
                        np.multiply(p_hat_all, (1.-m_hat_all))) \
            - theta * np.divide(d, p_hat_all)

    boot_theta, boot_t_stat = boot_manual(psi, J, smpls, se, weights, n_rep, dml_procedure)

    return boot_theta, boot_t_stat
