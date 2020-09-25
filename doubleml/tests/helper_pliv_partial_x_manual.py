import numpy as np
from sklearn.linear_model import LinearRegression

from doubleml.tests.helper_boot import boot_manual


def fit_nuisance_pliv_partial_x(Y, X, D, Z, ml_m, ml_g, ml_r, smpls):
    assert Z.ndim == 2
    g_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        g_hat.append(ml_g.fit(X[train_index], Y[train_index]).predict(X[test_index]))
    
    m_hat = []
    m_hat_array = np.zeros_like(Z)
    for i_instr in range(Z.shape[1]):
        this_instr_m_hat = []
        for idx, (train_index, test_index) in enumerate(smpls):
            this_instr_m_hat.append(ml_m.fit(X[train_index], Z[train_index, i_instr]).predict(X[test_index]))
            m_hat_array[test_index, i_instr] = this_instr_m_hat[idx]
        m_hat.append(this_instr_m_hat)
    
    r_hat = []
    r_hat_tilde = []
    for idx, (train_index, test_index) in enumerate(smpls):
        r_hat.append(ml_r.fit(X[train_index], D[train_index]).predict(X[test_index]))
        r_hat_tilde.append(LinearRegression(fit_intercept=True).fit(m_hat_array[train_index],
                                                                    r_hat[idx]).predict(m_hat_array[test_index]))
    
    return g_hat, r_hat, r_hat_tilde


def pliv_partial_x_dml1(Y, X, D, Z, g_hat, r_hat, r_hat_tilde, smpls, score):
    thetas = np.zeros(len(smpls))
    n_obs = len(Y)
    
    for idx, (train_index, test_index) in enumerate(smpls):
        u_hat = Y[test_index] - g_hat[idx]
        w_hat = D[test_index] - r_hat[idx]
        thetas[idx] = pliv_partial_x_orth(u_hat, w_hat, r_hat_tilde[idx], D[test_index], score)
    theta_hat = np.mean(thetas)
    
    ses = np.zeros(len(smpls))
    for idx, (train_index, test_index) in enumerate(smpls):
        u_hat = Y[test_index] - g_hat[idx]
        w_hat = D[test_index] - r_hat[idx]
        ses[idx] = var_pliv_partial_x(theta_hat, D[test_index],
                                      u_hat, w_hat, r_hat_tilde[idx],
                                      score, n_obs)
    se = np.sqrt(np.mean(ses))
    
    return theta_hat, se


def pliv_partial_x_dml2(Y, X, D, Z, g_hat, r_hat, r_hat_tilde, smpls, score):
    n_obs = len(Y)
    u_hat = np.zeros_like(Y)
    w_hat = np.zeros_like(D)
    r_hat_tilde_array = np.zeros_like(D)
    for idx, (train_index, test_index) in enumerate(smpls):
        u_hat[test_index] = Y[test_index] - g_hat[idx]
        w_hat[test_index] = D[test_index] - r_hat[idx]
        r_hat_tilde_array[test_index] = r_hat_tilde[idx]
    theta_hat = pliv_partial_x_orth(u_hat, w_hat, r_hat_tilde_array, D, score)
    se = np.sqrt(var_pliv_partial_x(theta_hat, D, u_hat, w_hat, r_hat_tilde_array, score, n_obs))
    
    return theta_hat, se


def var_pliv_partial_x(theta, d, u_hat, w_hat, r_hat_tilde, score, n_obs):
    if score == 'partialling out':
        var = 1/n_obs * 1/np.power(np.mean(np.multiply(r_hat_tilde, w_hat)), 2) * \
              np.mean(np.power(np.multiply(u_hat - w_hat*theta, r_hat_tilde), 2))
    else:
        raise ValueError('invalid score')
    
    return var


def pliv_partial_x_orth(u_hat, w_hat, r_hat_tilde, D, score):
    if score == 'partialling out':
        res = np.mean(np.multiply(r_hat_tilde, u_hat))/np.mean(np.multiply(r_hat_tilde, w_hat))
    else:
      raise ValueError('invalid score')
    
    return res


def boot_pliv_partial_x(theta, Y, D, Z, g_hat, r_hat, r_hat_tilde, smpls, score, se, bootstrap, n_rep, dml_procedure):
    u_hat = np.zeros_like(Y)
    w_hat = np.zeros_like(D)
    r_hat_tilde_array = np.zeros_like(D)
    n_folds = len(smpls)
    J = np.zeros(n_folds)
    for idx, (train_index, test_index) in enumerate(smpls):
        u_hat[test_index] = Y[test_index] - g_hat[idx]
        w_hat[test_index] = D[test_index] - r_hat[idx]
        r_hat_tilde_array[test_index] = r_hat_tilde[idx]
        if dml_procedure == 'dml1':
            if score == 'partialling out':
                J[idx] = np.mean(-np.multiply(r_hat_tilde_array[test_index], w_hat[test_index]))

    if dml_procedure == 'dml2':
        if score == 'partialling out':
            J = np.mean(-np.multiply(r_hat_tilde_array, w_hat))

    if score == 'partialling out':
        psi = np.multiply(u_hat - w_hat*theta, r_hat_tilde_array)
    else:
        raise ValueError('invalid score')
    
    boot_theta = boot_manual(psi, J, smpls, se, bootstrap, n_rep, dml_procedure)

    return boot_theta
