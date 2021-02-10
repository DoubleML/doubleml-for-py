import numpy as np
from sklearn.model_selection import KFold, GridSearchCV

from ._utils_boot import boot_manual, draw_weights


def fit_nuisance_pliv(y, x, d, z, ml_m, ml_g, ml_r, smpls, g_params=None, m_params=None, r_params=None):
    g_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if g_params is not None:
            ml_g.set_params(**g_params[idx])
        g_hat.append(ml_g.fit(x[train_index], y[train_index]).predict(x[test_index]))

    m_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if m_params is not None:
            ml_m.set_params(**m_params[idx])
        m_hat.append(ml_m.fit(x[train_index], z[train_index]).predict(x[test_index]))

    r_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if r_params is not None:
            ml_r.set_params(**r_params[idx])
        r_hat.append(ml_r.fit(x[train_index], d[train_index]).predict(x[test_index]))

    return g_hat, m_hat, r_hat


def tune_nuisance_pliv(y, x, d, z, ml_m, ml_g, ml_r, smpls, n_folds_tune, param_grid_g, param_grid_m, param_grid_r):
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
        m_tune_res[idx] = m_grid_search.fit(x[train_index, :], z[train_index])

    r_tune_res = [None] * len(smpls)
    for idx, (train_index, _) in enumerate(smpls):
        r_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
        r_grid_search = GridSearchCV(ml_r, param_grid_r,
                                     cv=r_tune_resampling)
        r_tune_res[idx] = r_grid_search.fit(x[train_index, :], d[train_index])

    g_best_params = [xx.best_params_ for xx in g_tune_res]
    m_best_params = [xx.best_params_ for xx in m_tune_res]
    r_best_params = [xx.best_params_ for xx in r_tune_res]

    return g_best_params, m_best_params, r_best_params


def pliv_dml1(y, x, d, z, g_hat, m_hat, r_hat, smpls, score):
    thetas = np.zeros(len(smpls))
    n_obs = len(y)

    for idx, (_, test_index) in enumerate(smpls):
        u_hat = y[test_index] - g_hat[idx]
        v_hat = z[test_index] - m_hat[idx]
        w_hat = d[test_index] - r_hat[idx]
        thetas[idx] = pliv_orth(u_hat, v_hat, w_hat, d[test_index], score)
    theta_hat = np.mean(thetas)

    u_hat = np.zeros_like(y, dtype='float64')
    v_hat = np.zeros_like(z, dtype='float64')
    w_hat = np.zeros_like(d, dtype='float64')
    for idx, (_, test_index) in enumerate(smpls):
        u_hat[test_index] = y[test_index] - g_hat[idx]
        v_hat[test_index] = z[test_index] - m_hat[idx]
        w_hat[test_index] = d[test_index] - r_hat[idx]
    se = np.sqrt(var_pliv(theta_hat, d, u_hat, v_hat, w_hat, score, n_obs))

    return theta_hat, se


def pliv_dml2(y, x, d, z, g_hat, m_hat, r_hat, smpls, score):
    n_obs = len(y)
    u_hat = np.zeros_like(y, dtype='float64')
    v_hat = np.zeros_like(z, dtype='float64')
    w_hat = np.zeros_like(d, dtype='float64')
    for idx, (_, test_index) in enumerate(smpls):
        u_hat[test_index] = y[test_index] - g_hat[idx]
        v_hat[test_index] = z[test_index] - m_hat[idx]
        w_hat[test_index] = d[test_index] - r_hat[idx]
    theta_hat = pliv_orth(u_hat, v_hat, w_hat, d, score)
    se = np.sqrt(var_pliv(theta_hat, d, u_hat, v_hat, w_hat, score, n_obs))

    return theta_hat, se


def var_pliv(theta, d, u_hat, v_hat, w_hat, score, n_obs):
    assert score == 'partialling out'
    var = 1/n_obs * 1/np.power(np.mean(np.multiply(v_hat, w_hat)), 2) * \
        np.mean(np.power(np.multiply(u_hat - w_hat*theta, v_hat), 2))

    return var


def pliv_orth(u_hat, v_hat, w_hat, d, score):
    assert score == 'partialling out'
    res = np.mean(np.multiply(v_hat, u_hat))/np.mean(np.multiply(v_hat, w_hat))

    return res


def boot_pliv(theta, y, d, z, g_hat, m_hat, r_hat, smpls, score, se, bootstrap, n_rep, dml_procedure):
    n_obs = len(y)
    weights = draw_weights(bootstrap, n_rep, n_obs)
    assert np.isscalar(theta)
    boot_theta, boot_t_stat = boot_pliv_single_treat(theta, y, d, z, g_hat, m_hat, r_hat,
                                                     smpls, score, se, weights, n_rep, dml_procedure)
    return boot_theta, boot_t_stat


def boot_pliv_single_treat(theta, y, d, z, g_hat, m_hat, r_hat, smpls, score, se, weights, n_rep, dml_procedure):
    assert score == 'partialling out'
    u_hat = np.zeros_like(y, dtype='float64')
    v_hat = np.zeros_like(z, dtype='float64')
    w_hat = np.zeros_like(d, dtype='float64')
    n_folds = len(smpls)
    J = np.zeros(n_folds)
    for idx, (_, test_index) in enumerate(smpls):
        u_hat[test_index] = y[test_index] - g_hat[idx]
        v_hat[test_index] = z[test_index] - m_hat[idx]
        w_hat[test_index] = d[test_index] - r_hat[idx]
        if dml_procedure == 'dml1':
            J[idx] = np.mean(-np.multiply(v_hat[test_index], w_hat[test_index]))

    if dml_procedure == 'dml2':
        J = np.mean(-np.multiply(v_hat, w_hat))

    psi = np.multiply(u_hat - w_hat*theta, v_hat)

    boot_theta, boot_t_stat = boot_manual(psi, J, smpls, se, weights, n_rep, dml_procedure)

    return boot_theta, boot_t_stat
