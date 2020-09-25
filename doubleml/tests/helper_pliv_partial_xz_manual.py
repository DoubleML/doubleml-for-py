import numpy as np

from doubleml.tests.helper_boot import boot_manual


def fit_nuisance_pliv_partial_xz(Y, X, D, Z, ml_m, ml_g, ml_r, smpls):
    g_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        g_hat.append(ml_g.fit(X[train_index], Y[train_index]).predict(X[test_index]))

    XZ = np.hstack((X, Z))
    m_hat = []
    m_hat_vector = np.zeros_like(D)
    for idx, (train_index, test_index) in enumerate(smpls):
        m_hat.append(ml_m.fit(XZ[train_index], D[train_index]).predict(XZ[test_index]))
        m_hat_vector[test_index] = m_hat[idx]
    
    m_hat_tilde = []
    for idx, (train_index, test_index) in enumerate(smpls):
        m_hat_tilde.append(ml_r.fit(X[train_index], m_hat_vector[train_index]).predict(X[test_index]))
    
    return g_hat, m_hat, m_hat_tilde


def pliv_partial_xz_dml1(Y, X, D, Z, g_hat, m_hat, m_hat_tilde, smpls, score):
    thetas = np.zeros(len(smpls))
    n_obs = len(Y)
    
    for idx, (train_index, test_index) in enumerate(smpls):
        u_hat = Y[test_index] - g_hat[idx]
        v_hat = m_hat[idx] - m_hat_tilde[idx]
        w_hat = D[test_index] - m_hat_tilde[idx]
        thetas[idx] = pliv_partial_xz_orth(u_hat, v_hat, w_hat, D[test_index], score)
    theta_hat = np.mean(thetas)
    
    ses = np.zeros(len(smpls))
    for idx, (train_index, test_index) in enumerate(smpls):
        u_hat = Y[test_index] - g_hat[idx]
        v_hat = m_hat[idx] - m_hat_tilde[idx]
        w_hat = D[test_index] - m_hat_tilde[idx]
        ses[idx] = var_pliv_partial_xz(theta_hat, D[test_index],
                                       u_hat, v_hat, w_hat,
                                       score, n_obs)
    se = np.sqrt(np.mean(ses))
    
    return theta_hat, se


def pliv_partial_xz_dml2(Y, X, D, Z, g_hat, m_hat, m_hat_tilde, smpls, score):
    thetas = np.zeros(len(smpls))
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
    
    boot_theta = boot_manual(psi, J, smpls, se, bootstrap, n_rep, dml_procedure)

    return boot_theta
