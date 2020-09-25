import numpy as np

from doubleml.tests.helper_boot import boot_manual

def fit_nuisance_irm(Y, X, D, ml_m, ml_g, smpls, score):
    g_hat0 = []
    g_hat1 = []
    for idx, (train_index, test_index) in enumerate(smpls):
        train_index0 =np.intersect1d(np.where(D==0)[0], train_index)
        g_hat0.append(ml_g.fit(X[train_index0],Y[train_index0]).predict(X[test_index]))
    
    if score == 'ATE':
        for idx, (train_index, test_index) in enumerate(smpls):
            train_index1 =np.intersect1d(np.where(D==1)[0], train_index)
            g_hat1.append(ml_g.fit(X[train_index1],Y[train_index1]).predict(X[test_index]))
    else:
        for idx, (train_index, test_index) in enumerate(smpls):
            # fill it up, but its not further used
            g_hat1.append(np.zeros_like(g_hat0[idx]))
    
    m_hat = []
    p_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        m_hat.append(ml_m.fit(X[train_index],D[train_index]).predict_proba(X[test_index])[:, 1])
        p_hat.append(np.mean(D[test_index]))
    
    return g_hat0, g_hat1, m_hat, p_hat

def irm_dml1(Y, X, D, g_hat0, g_hat1, m_hat, p_hat, smpls, score):
    thetas = np.zeros(len(smpls))
    n_obs = len(Y)
    
    for idx, (train_index, test_index) in enumerate(smpls):
        u_hat0 = Y[test_index] - g_hat0[idx]
        u_hat1 = Y[test_index] - g_hat1[idx]
        thetas[idx] = irm_orth(g_hat0[idx], g_hat1[idx],
                               m_hat[idx], p_hat[idx],
                               u_hat0, u_hat1,
                               D[test_index], score)
    theta_hat = np.mean(thetas)
    
    ses = np.zeros(len(smpls))
    for idx, (train_index, test_index) in enumerate(smpls):
        u_hat0 = Y[test_index] - g_hat0[idx]
        u_hat1 = Y[test_index] - g_hat1[idx]
        ses[idx] = var_irm(theta_hat, g_hat0[idx], g_hat1[idx],
                           m_hat[idx], p_hat[idx],
                           u_hat0, u_hat1,
                           D[test_index], score, n_obs)
    se = np.sqrt(np.mean(ses))
    
    return theta_hat, se

def irm_dml2(Y, X, D, g_hat0, g_hat1, m_hat, p_hat, smpls, score):
    n_obs = len(Y)
    u_hat0 = np.zeros_like(Y)
    u_hat1 = np.zeros_like(Y)
    g_hat0_all = np.zeros_like(Y)
    g_hat1_all = np.zeros_like(Y)
    m_hat_all = np.zeros_like(Y)
    p_hat_all = np.zeros_like(Y)
    for idx, (train_index, test_index) in enumerate(smpls):
        u_hat0[test_index] = Y[test_index] - g_hat0[idx]
        u_hat1[test_index] = Y[test_index] - g_hat1[idx]
        g_hat0_all[test_index] = g_hat0[idx]
        g_hat1_all[test_index] = g_hat1[idx]
        m_hat_all[test_index] = m_hat[idx]
        p_hat_all[test_index] = p_hat[idx]
    theta_hat = irm_orth(g_hat0_all, g_hat1_all, m_hat_all, p_hat_all,
                         u_hat0, u_hat1, D, score)
    se = np.sqrt(var_irm(theta_hat, g_hat0_all, g_hat1_all,
                         m_hat_all, p_hat_all,
                         u_hat0, u_hat1,
                         D, score, n_obs))
    
    return theta_hat, se
    
def var_irm(theta, g_hat0, g_hat1, m_hat, p_hat, u_hat0, u_hat1, D, score, n_obs):
    if score == 'ATE':
        var = 1/n_obs * np.mean(np.power(g_hat1 - g_hat0 \
                      + np.divide(np.multiply(D, u_hat1), m_hat) \
                      - np.divide(np.multiply(1.-D, u_hat0), 1.-m_hat) - theta, 2))
    elif score == 'ATTE':
        var = 1/n_obs * np.mean(np.power(np.divide(np.multiply(D, u_hat0), p_hat) \
                      - np.divide(np.multiply(m_hat, np.multiply(1.-D, u_hat0)),
                                  np.multiply(p_hat, (1.-m_hat))) \
                      - theta * np.divide(D, p_hat), 2)) \
              / np.power(np.mean(np.divide(D, p_hat)), 2)
    else:
        raise ValueError('invalid score')
    
    return var

def irm_orth(g_hat0, g_hat1, m_hat, p_hat, u_hat0, u_hat1, D, score):
    if score == 'ATE':
        res = np.mean(g_hat1 - g_hat0 \
                      + np.divide(np.multiply(D, u_hat1), m_hat) \
                      - np.divide(np.multiply(1.-D, u_hat0), 1.-m_hat))
    elif score == 'ATTE':
        res = np.mean(np.divide(np.multiply(D, u_hat0), p_hat) \
                      - np.divide(np.multiply(m_hat, np.multiply(1.-D, u_hat0)),
                                  np.multiply(p_hat, (1.-m_hat)))) \
              / np.mean(np.divide(D, p_hat))
    
    return res

def boot_irm(theta, Y, D, g_hat0, g_hat1, m_hat, p_hat, smpls, score, se, bootstrap, n_rep, dml_procedure):
    u_hat0 = np.zeros_like(Y)
    u_hat1 = np.zeros_like(Y)
    g_hat0_all = np.zeros_like(Y)
    g_hat1_all = np.zeros_like(Y)
    m_hat_all = np.zeros_like(Y)
    p_hat_all = np.zeros_like(Y)
    n_folds = len(smpls)
    J = np.zeros(n_folds)
    for idx, (train_index, test_index) in enumerate(smpls):
        u_hat0[test_index] = Y[test_index] - g_hat0[idx]
        u_hat1[test_index] = Y[test_index] - g_hat1[idx]
        g_hat0_all[test_index] = g_hat0[idx]
        g_hat1_all[test_index] = g_hat1[idx]
        m_hat_all[test_index] = m_hat[idx]
        p_hat_all[test_index] = p_hat[idx]
        if dml_procedure == 'dml1':
            if score == 'ATE':
                J[idx] = -1.0
            elif score == 'ATTE':
                J[idx] = np.mean(-np.divide(D[test_index], p_hat_all[test_index]))

    if dml_procedure == 'dml2':
        if score == 'ATE':
            J = -1.0
        elif score == 'ATTE':
            J = np.mean(-np.divide(D, p_hat_all))
    
    if score == 'ATE':
        psi = g_hat1_all - g_hat0_all \
                + np.divide(np.multiply(D, u_hat1), m_hat_all) \
                - np.divide(np.multiply(1.-D, u_hat0), 1.-m_hat_all) - theta
    elif score == 'ATTE':
        psi = np.divide(np.multiply(D, u_hat0), p_hat_all) \
                - np.divide(np.multiply(m_hat_all, np.multiply(1.-D, u_hat0)),
                            np.multiply(p_hat_all, (1.-m_hat_all))) \
                - theta * np.divide(D, p_hat_all)
    else:
        raise ValueError('invalid score')

    boot_theta = boot_manual(psi, J, smpls, se, bootstrap, n_rep, dml_procedure)
    
    return boot_theta
