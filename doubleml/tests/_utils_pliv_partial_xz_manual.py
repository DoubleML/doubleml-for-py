import numpy as np
from sklearn.model_selection import KFold, GridSearchCV

from ._utils_boot import boot_manual, draw_weights


def fit_nuisance_pliv_partial_xz(y, x, d, z, ml_m, ml_g, ml_r, smpls, g_params=None, m_params=None, r_params=None):
    g_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if g_params is not None:
            ml_g.set_params(**g_params[idx])
        g_hat.append(ml_g.fit(x[train_index], y[train_index]).predict(x[test_index]))

    xz = np.hstack((x, z))
    m_hat = []
    m_hat_train = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if m_params is not None:
            ml_m.set_params(**m_params[idx])
        ml_m.fit(xz[train_index], d[train_index])
        m_hat.append(ml_m.predict(xz[test_index]))
        m_hat_train.append(ml_m.predict(xz[train_index]))

    m_hat_tilde = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if r_params is not None:
            ml_r.set_params(**r_params[idx])
        m_hat_tilde.append(ml_r.fit(x[train_index], m_hat_train[idx]).predict(x[test_index]))

    return g_hat, m_hat, m_hat_tilde


def tune_nuisance_pliv_partial_xz(y, x, d, z, ml_m, ml_g, ml_r, smpls, n_folds_tune, param_grid_g, param_grid_m, param_grid_r):
    xz = np.hstack((x, z))
    g_tune_res = [None] * len(smpls)
    for idx, (train_index, _) in enumerate(smpls):
        g_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
        g_grid_search = GridSearchCV(ml_g, param_grid_g,
                                     cv=g_tune_resampling)
        g_tune_res[idx] = g_grid_search.fit(x[train_index, :], y[train_index])

    m_tune_res = [None] * len(smpls)
    for idx, (train_index, _) in enumerate(smpls):
        m_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
        m_grid_search = GridSearchCV(ml_m, param_grid_m,
                                     cv=m_tune_resampling)
        m_tune_res[idx] = m_grid_search.fit(xz[train_index, :], d[train_index])

    r_tune_res = [None] * len(smpls)
    for idx, (train_index, _) in enumerate(smpls):
        m_hat = m_tune_res[idx].predict(xz[train_index, :])
        r_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
        r_grid_search = GridSearchCV(ml_r, param_grid_r,
                                     cv=r_tune_resampling)
        r_tune_res[idx] = r_grid_search.fit(x[train_index, :], m_hat)

    g_best_params = [xx.best_params_ for xx in g_tune_res]
    m_best_params = [xx.best_params_ for xx in m_tune_res]
    r_best_params = [xx.best_params_ for xx in r_tune_res]

    return g_best_params, m_best_params, r_best_params


def pliv_partial_xz_dml1(y, x, d, z, g_hat, m_hat, m_hat_tilde, smpls, score):
    thetas = np.zeros(len(smpls))
    n_obs = len(y)

    for idx, (_, test_index) in enumerate(smpls):
        u_hat = y[test_index] - g_hat[idx]
        v_hat = m_hat[idx] - m_hat_tilde[idx]
        w_hat = d[test_index] - m_hat_tilde[idx]
        thetas[idx] = pliv_partial_xz_orth(u_hat, v_hat, w_hat, d[test_index], score)
    theta_hat = np.mean(thetas)

    u_hat = np.zeros_like(y, dtype='float64')
    v_hat = np.zeros_like(d, dtype='float64')
    w_hat = np.zeros_like(d, dtype='float64')
    for idx, (_, test_index) in enumerate(smpls):
        u_hat[test_index] = y[test_index] - g_hat[idx]
        v_hat[test_index] = m_hat[idx] - m_hat_tilde[idx]
        w_hat[test_index] = d[test_index] - m_hat_tilde[idx]
    se = np.sqrt(var_pliv_partial_xz(theta_hat, d, u_hat, v_hat, w_hat, score, n_obs))

    return theta_hat, se


def pliv_partial_xz_dml2(y, x, d, z, g_hat, m_hat, m_hat_tilde, smpls, score):
    n_obs = len(y)
    u_hat = np.zeros_like(y, dtype='float64')
    v_hat = np.zeros_like(d, dtype='float64')
    w_hat = np.zeros_like(d, dtype='float64')
    for idx, (_, test_index) in enumerate(smpls):
        u_hat[test_index] = y[test_index] - g_hat[idx]
        v_hat[test_index] = m_hat[idx] - m_hat_tilde[idx]
        w_hat[test_index] = d[test_index] - m_hat_tilde[idx]
    theta_hat = pliv_partial_xz_orth(u_hat, v_hat, w_hat, d, score)
    se = np.sqrt(var_pliv_partial_xz(theta_hat, d, u_hat, v_hat, w_hat, score, n_obs))

    return theta_hat, se


def var_pliv_partial_xz(theta, d, u_hat, v_hat, w_hat, score, n_obs):
    assert score == 'partialling out'
    var = 1/n_obs * 1/np.power(np.mean(np.multiply(v_hat, w_hat)), 2) * \
        np.mean(np.power(np.multiply(u_hat - w_hat*theta, v_hat), 2))

    return var


def pliv_partial_xz_orth(u_hat, v_hat, w_hat, d, score):
    assert score == 'partialling out'
    res = np.mean(np.multiply(v_hat, u_hat))/np.mean(np.multiply(v_hat, w_hat))

    return res


def boot_pliv_partial_xz(theta, y, d, z, g_hat, m_hat, m_hat_tilde, smpls, score, se, bootstrap, n_rep, dml_procedure):
    n_obs = len(y)
    weights = draw_weights(bootstrap, n_rep, n_obs)
    assert np.isscalar(theta)
    boot_theta, boot_t_stat = boot_pliv_partial_xz_single_treat(theta, y, d, z, g_hat, m_hat, m_hat_tilde,
                                                                smpls, score, se, weights, n_rep, dml_procedure)
    return boot_theta, boot_t_stat


def boot_pliv_partial_xz_single_treat(theta, y, d, z, g_hat, m_hat, m_hat_tilde, smpls, score, se, weights,
                                      n_rep, dml_procedure):
    assert score == 'partialling out'
    u_hat = np.zeros_like(y, dtype='float64')
    v_hat = np.zeros_like(d, dtype='float64')
    w_hat = np.zeros_like(d, dtype='float64')
    n_folds = len(smpls)
    J = np.zeros(n_folds)
    for idx, (_, test_index) in enumerate(smpls):
        u_hat[test_index] = y[test_index] - g_hat[idx]
        v_hat[test_index] = m_hat[idx] - m_hat_tilde[idx]
        w_hat[test_index] = d[test_index] - m_hat_tilde[idx]
        if dml_procedure == 'dml1':
            J[idx] = np.mean(-np.multiply(v_hat[test_index], w_hat[test_index]))

    if dml_procedure == 'dml2':
        J = np.mean(-np.multiply(v_hat, w_hat))

    psi = np.multiply(u_hat - w_hat*theta, v_hat)

    boot_theta, boot_t_stat = boot_manual(psi, J, smpls, se, weights, n_rep, dml_procedure)

    return boot_theta, boot_t_stat
