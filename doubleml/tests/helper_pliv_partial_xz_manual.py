import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.base import clone

from doubleml.tests.helper_boot import boot_manual


def fit_nuisance_pliv_partial_xz(Y, X, D, Z, ml_m, ml_g, ml_r, smpls, g_params=None, m_params=None, r_params=None):
    g_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if g_params is not None:
            ml_g.set_params(**g_params[idx])
        g_hat.append(ml_g.fit(X[train_index], Y[train_index]).predict(X[test_index]))

    XZ = np.hstack((X, Z))
    m_hat = []
    m_hat_train = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if m_params is not None:
            ml_m.set_params(**m_params[idx])
        ml_m.fit(XZ[train_index], D[train_index])
        m_hat.append(ml_m.predict(XZ[test_index]))
        m_hat_train.append(ml_m.predict(XZ[train_index]))
    
    m_hat_tilde = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if r_params is not None:
            ml_r.set_params(**r_params[idx])
        m_hat_tilde.append(ml_r.fit(X[train_index], m_hat_train[idx]).predict(X[test_index]))
    
    return g_hat, m_hat, m_hat_tilde


def tune_nuisance_pliv_partial_xz(Y, X, D, Z, ml_m, ml_g, ml_r, smpls, n_folds_tune, param_grid_g, param_grid_m, param_grid_r):
    XZ = np.hstack((X, Z))
    g_tune_res = [None] * len(smpls)
    for idx, (train_index, test_index) in enumerate(smpls):
        g_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
        g_grid_search = GridSearchCV(ml_g, param_grid_g,
                                     cv=g_tune_resampling)
        g_tune_res[idx] = g_grid_search.fit(X[train_index, :], Y[train_index])

    m_tune_res = [None] * len(smpls)
    for idx, (train_index, test_index) in enumerate(smpls):
        m_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
        m_grid_search = GridSearchCV(ml_m, param_grid_m,
                                     cv=m_tune_resampling)
        m_tune_res[idx] = m_grid_search.fit(XZ[train_index, :], D[train_index])

    r_tune_res = [None] * len(smpls)
    for idx, (train_index, test_index) in enumerate(smpls):
        m_hat = m_tune_res[idx].predict(XZ[train_index, :])
        r_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
        r_grid_search = GridSearchCV(ml_r, param_grid_r,
                                     cv=r_tune_resampling)
        r_tune_res[idx] = r_grid_search.fit(X[train_index, :], m_hat)

    g_best_params = [xx.best_params_ for xx in g_tune_res]
    m_best_params = [xx.best_params_ for xx in m_tune_res]
    r_best_params = [xx.best_params_ for xx in r_tune_res]

    return g_best_params, m_best_params, r_best_params


def pliv_partial_xz_dml1(Y, X, D, Z, g_hat, m_hat, m_hat_tilde, smpls, score):
    thetas = np.zeros(len(smpls))
    n_obs = len(Y)
    
    for idx, (train_index, test_index) in enumerate(smpls):
        u_hat = Y[test_index] - g_hat[idx]
        v_hat = m_hat[idx] - m_hat_tilde[idx]
        w_hat = D[test_index] - m_hat_tilde[idx]
        thetas[idx] = pliv_partial_xz_orth(u_hat, v_hat, w_hat, D[test_index], score)
    theta_hat = np.mean(thetas)
    
    u_hat = np.zeros_like(Y)
    v_hat = np.zeros_like(D)
    w_hat = np.zeros_like(D)
    for idx, (train_index, test_index) in enumerate(smpls):
        u_hat[test_index] = Y[test_index] - g_hat[idx]
        v_hat[test_index] = m_hat[idx] - m_hat_tilde[idx]
        w_hat[test_index] = D[test_index] - m_hat_tilde[idx]
    se = np.sqrt(var_pliv_partial_xz(theta_hat, D, u_hat, v_hat, w_hat, score, n_obs))
    
    return theta_hat, se


def pliv_partial_xz_dml2(Y, X, D, Z, g_hat, m_hat, m_hat_tilde, smpls, score):
    n_obs = len(Y)
    u_hat = np.zeros_like(Y)
    v_hat = np.zeros_like(D)
    w_hat = np.zeros_like(D)
    for idx, (train_index, test_index) in enumerate(smpls):
        u_hat[test_index] = Y[test_index] - g_hat[idx]
        v_hat[test_index] = m_hat[idx] - m_hat_tilde[idx]
        w_hat[test_index] = D[test_index] - m_hat_tilde[idx]
    theta_hat = pliv_partial_xz_orth(u_hat, v_hat, w_hat, D, score)
    se = np.sqrt(var_pliv_partial_xz(theta_hat, D, u_hat, v_hat, w_hat, score, n_obs))
    
    return theta_hat, se


def var_pliv_partial_xz(theta, d, u_hat, v_hat, w_hat, score, n_obs):
    if score == 'partialling out':
        var = 1/n_obs * 1/np.power(np.mean(np.multiply(v_hat, w_hat)), 2) * \
              np.mean(np.power(np.multiply(u_hat - w_hat*theta, v_hat), 2))
    else:
        raise ValueError('invalid score')
    
    return var


def pliv_partial_xz_orth(u_hat, v_hat, w_hat, D, score):
    if score == 'partialling out':
        res = np.mean(np.multiply(v_hat, u_hat))/np.mean(np.multiply(v_hat, w_hat))
    else:
      raise ValueError('invalid score')
    
    return res


def boot_pliv_partial_xz(theta, Y, D, Z, g_hat, m_hat, m_hat_tilde, smpls, score, se, bootstrap, n_rep, dml_procedure):
    u_hat = np.zeros_like(Y)
    v_hat = np.zeros_like(D)
    w_hat = np.zeros_like(D)
    n_folds = len(smpls)
    J = np.zeros(n_folds)
    for idx, (train_index, test_index) in enumerate(smpls):
        u_hat[test_index] = Y[test_index] - g_hat[idx]
        v_hat[test_index] = m_hat[idx] - m_hat_tilde[idx]
        w_hat[test_index] = D[test_index] - m_hat_tilde[idx]
        if dml_procedure == 'dml1':
            if score == 'partialling out':
                J[idx] = np.mean(-np.multiply(v_hat[test_index], w_hat[test_index]))

    if dml_procedure == 'dml2':
        if score == 'partialling out':
            J = np.mean(-np.multiply(v_hat, w_hat))

    if score == 'partialling out':
        psi = np.multiply(u_hat - w_hat*theta, v_hat)
    else:
        raise ValueError('invalid score')
    
    boot_theta, boot_t_stat = boot_manual(psi, J, smpls, se, bootstrap, n_rep, dml_procedure)

    return boot_theta, boot_t_stat
