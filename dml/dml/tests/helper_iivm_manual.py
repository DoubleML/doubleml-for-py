import numpy as np


def fit_nuisance_iivm(Y, X, D, Z, ml_m, ml_g, ml_r, smpls):
    g_hat0 = []
    for idx, (train_index, test_index) in enumerate(smpls):
        train_index0 =np.intersect1d(np.where(Z==0)[0], train_index)
        g_hat0.append(ml_g.fit(X[train_index0], Y[train_index0]).predict(X[test_index]))
    
    g_hat1 = []
    for idx, (train_index, test_index) in enumerate(smpls):
        train_index1 =np.intersect1d(np.where(Z==1)[0], train_index)
        g_hat1.append(ml_g.fit(X[train_index1], Y[train_index1]).predict(X[test_index]))
    
    m_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        m_hat.append(ml_m.fit(X[train_index], Z[train_index]).predict_proba(X[test_index])[:, 1])
    
    r_hat0 = []
    for idx, (train_index, test_index) in enumerate(smpls):
        train_index0 =np.intersect1d(np.where(Z==0)[0], train_index)
        r_hat0.append(ml_r.fit(X[train_index0], D[train_index0]).predict_proba(X[test_index])[:, 1])
    
    r_hat1 = []
    for idx, (train_index, test_index) in enumerate(smpls):
        train_index1 =np.intersect1d(np.where(Z==1)[0], train_index)
        r_hat1.append(ml_r.fit(X[train_index1], D[train_index1]).predict_proba(X[test_index])[:, 1])
    
    
    return g_hat0, g_hat1, m_hat, r_hat0, r_hat1

def iivm_dml1(Y, X, D, Z, g_hat0, g_hat1, m_hat, r_hat0, r_hat1, smpls, inf_model):
    thetas = np.zeros(len(smpls))
    n_obs = len(Y)
    
    for idx, (train_index, test_index) in enumerate(smpls):
        u_hat0 = Y[test_index] - g_hat0[idx]
        u_hat1 = Y[test_index] - g_hat1[idx]
        w_hat0 = D[test_index] - r_hat0[idx]
        w_hat1 = D[test_index] - r_hat1[idx]
        thetas[idx] = iivm_orth(g_hat0[idx], g_hat1[idx],
                                m_hat[idx],
                                r_hat0[idx], r_hat1[idx],
                                u_hat0, u_hat1,
                                w_hat0, w_hat1,
                                Z[test_index], inf_model)
    theta_hat = np.mean(thetas)
    
    ses = np.zeros(len(smpls))
    for idx, (train_index, test_index) in enumerate(smpls):
        u_hat0 = Y[test_index] - g_hat0[idx]
        u_hat1 = Y[test_index] - g_hat1[idx]
        w_hat0 = D[test_index] - r_hat0[idx]
        w_hat1 = D[test_index] - r_hat1[idx]
        ses[idx] = var_iivm(theta_hat, g_hat0[idx], g_hat1[idx],
                            m_hat[idx],
                            r_hat0[idx], r_hat1[idx],
                            u_hat0, u_hat1,
                            w_hat0, w_hat1,
                            Z[test_index], inf_model, n_obs)
    se = np.sqrt(np.mean(ses))
    
    return theta_hat, se

def iivm_dml2(Y, X, D, Z, g_hat0, g_hat1, m_hat, r_hat0, r_hat1, smpls, inf_model):
    n_obs = len(Y)
    u_hat0 = np.zeros_like(Y)
    u_hat1 = np.zeros_like(Y)
    w_hat0 = np.zeros_like(Y)
    w_hat1 = np.zeros_like(Y)
    g_hat0_all = np.zeros_like(Y)
    g_hat1_all = np.zeros_like(Y)
    r_hat0_all = np.zeros_like(Y)
    r_hat1_all = np.zeros_like(Y)
    m_hat_all = np.zeros_like(Y)
    for idx, (train_index, test_index) in enumerate(smpls):
        u_hat0[test_index] = Y[test_index] - g_hat0[idx]
        u_hat1[test_index] = Y[test_index] - g_hat1[idx]
        w_hat0[test_index] = D[test_index] - r_hat0[idx]
        w_hat1[test_index] = D[test_index] - r_hat1[idx]
        g_hat0_all[test_index] = g_hat0[idx]
        g_hat1_all[test_index] = g_hat1[idx]
        r_hat0_all[test_index] = r_hat0[idx]
        r_hat1_all[test_index] = r_hat1[idx]
        m_hat_all[test_index] = m_hat[idx]
    theta_hat = iivm_orth(g_hat0_all, g_hat1_all, m_hat_all, r_hat0_all, r_hat1_all,
                          u_hat0, u_hat1, w_hat0, w_hat1, Z, inf_model)
    se = np.sqrt(var_iivm(theta_hat, g_hat0_all, g_hat1_all,
                          m_hat_all, r_hat0_all, r_hat1_all,
                          u_hat0, u_hat1, w_hat0, w_hat1,
                          Z, inf_model, n_obs))
    
    return theta_hat, se
    
def var_iivm(theta, g_hat0, g_hat1, m_hat, r_hat0, r_hat1, u_hat0, u_hat1, w_hat0, w_hat1, Z, se_type, n_obs):
    if se_type == 'LATE':
        var = 1/n_obs * np.mean(np.power(g_hat1 - g_hat0 \
                            + np.divide(np.multiply(Z, u_hat1), m_hat) \
                            - np.divide(np.multiply(1.-Z, u_hat0), 1.-m_hat) \
                        -theta*(r_hat1 - r_hat0 \
                            + np.divide(np.multiply(Z, w_hat1), m_hat) \
                            - np.divide(np.multiply(1.-Z, w_hat0), 1.-m_hat)), 2)) \
                      / np.power(np.mean(r_hat1 - r_hat0 \
                                + np.divide(np.multiply(Z, w_hat1), m_hat) \
                                - np.divide(np.multiply(1.-Z, w_hat0), 1.-m_hat)), 2)
    else:
        raise ValueError('invalid se_type')
    
    return var

def iivm_orth(g_hat0, g_hat1, m_hat, r_hat0, r_hat1, u_hat0, u_hat1, w_hat0, w_hat1, Z, inf_model):
    
    if inf_model == 'LATE':
        res = np.mean(g_hat1 - g_hat0 \
                      + np.divide(np.multiply(Z, u_hat1), m_hat) \
                      - np.divide(np.multiply(1.-Z, u_hat0), 1.-m_hat)) \
              / np.mean(r_hat1 - r_hat0 \
                        + np.divide(np.multiply(Z, w_hat1), m_hat) \
                        - np.divide(np.multiply(1.-Z, w_hat0), 1.-m_hat))
    else:
        raise ValueError('invalid inf_model')
    
    return res

def boot_iivm(theta, Y, D, Z, g_hat0, g_hat1, m_hat, r_hat0, r_hat1, smpls, inf_model, se, bootstrap, n_rep, dml_procedure):
    u_hat0 = np.zeros_like(Y)
    u_hat1 = np.zeros_like(Y)
    w_hat0 = np.zeros_like(Y)
    w_hat1 = np.zeros_like(Y)
    g_hat0_all = np.zeros_like(Y)
    g_hat1_all = np.zeros_like(Y)
    r_hat0_all = np.zeros_like(Y)
    r_hat1_all = np.zeros_like(Y)
    m_hat_all = np.zeros_like(Y)
    n_folds = len(smpls)
    J = np.zeros(n_folds)
    for idx, (train_index, test_index) in enumerate(smpls):
        u_hat0[test_index] = Y[test_index] - g_hat0[idx]
        u_hat1[test_index] = Y[test_index] - g_hat1[idx]
        w_hat0[test_index] = D[test_index] - r_hat0[idx]
        w_hat1[test_index] = D[test_index] - r_hat1[idx]
        g_hat0_all[test_index] = g_hat0[idx]
        g_hat1_all[test_index] = g_hat1[idx]
        r_hat0_all[test_index] = r_hat0[idx]
        r_hat1_all[test_index] = r_hat1[idx]
        m_hat_all[test_index] = m_hat[idx]
        if dml_procedure == 'dml1':
            if inf_model == 'LATE':
                J[idx] = np.mean(-(r_hat1_all[test_index] - r_hat0_all[test_index] \
                              + np.divide(np.multiply(Z[test_index], w_hat1[test_index]), m_hat_all[test_index]) \
                              - np.divide(np.multiply(1. - Z[test_index], w_hat0[test_index]), 1. - m_hat_all[test_index])))

    if dml_procedure == 'dml2':
        if inf_model == 'LATE':
            J = np.mean(-(r_hat1_all - r_hat0_all \
                          + np.divide(np.multiply(Z, w_hat1), m_hat_all) \
                          - np.divide(np.multiply(1. - Z, w_hat0), 1. - m_hat_all)))

    if inf_model == 'LATE':
        score = g_hat1_all - g_hat0_all \
                + np.divide(np.multiply(Z, u_hat1), m_hat_all) \
                - np.divide(np.multiply(1.-Z, u_hat0), 1.-m_hat_all) \
                -theta*(r_hat1_all - r_hat0_all \
                    + np.divide(np.multiply(Z, w_hat1), m_hat_all) \
                    - np.divide(np.multiply(1.-Z, w_hat0), 1.-m_hat_all))
    else:
        raise ValueError('invalid inf_model')
    
    n_obs = len(score)
    boot_theta = np.zeros(n_rep)
    if bootstrap == 'wild':
        # if method wild for unit test comparability draw all rv at one step
        xx_sample = np.random.normal(loc=0.0, scale=1.0, size=(n_rep, n_obs))
        yy_sample = np.random.normal(loc=0.0, scale=1.0, size=(n_rep, n_obs))
    
    for i_rep in range(n_rep):
        if bootstrap == 'Bayes':
            weights = np.random.exponential(scale=1.0, size=n_obs) - 1.
        elif bootstrap == 'normal':
            weights = np.random.normal(loc=0.0, scale=1.0, size=n_obs)
        elif bootstrap == 'wild':
            xx = xx_sample[i_rep,:]
            yy = yy_sample[i_rep,:]
            weights = xx / np.sqrt(2) + (np.power(yy,2) - 1)/2
        else:
            raise ValueError('invalid bootstrap method')

        if dml_procedure == 'dml1':
            this_boot_theta = np.zeros(n_folds)
            for idx, (train_index, test_index) in enumerate(smpls):
                this_boot_theta[idx] = np.mean(np.multiply(np.divide(weights[test_index], se),
                                                           score[test_index] / J[idx]))
            boot_theta[i_rep] = np.mean(this_boot_theta)
        elif dml_procedure == 'dml2':
            boot_theta[i_rep] = np.mean(np.multiply(np.divide(weights, se),
                                                    score / J))
    
    return boot_theta
