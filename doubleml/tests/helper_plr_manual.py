import numpy as np
import pytest
import math
import scipy


def fit_nuisance_plr(Y, X, D, ml_m, ml_g, smpls):
    g_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        g_hat.append(ml_g.fit(X[train_index],Y[train_index]).predict(X[test_index]))
    
    m_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        m_hat.append(ml_m.fit(X[train_index],D[train_index]).predict(X[test_index]))
    
    return g_hat, m_hat


def plr_dml1(Y, X, D, g_hat, m_hat, smpls, score, se_reestimate=False):
    thetas = np.zeros(len(smpls))
    n_obs = len(Y)
    
    for idx, (train_index, test_index) in enumerate(smpls):
        v_hat = D[test_index] - m_hat[idx]
        u_hat = Y[test_index] - g_hat[idx]
        thetas[idx] = plr_orth(v_hat, u_hat, D[test_index], score)
    theta_hat = np.mean(thetas)

    if se_reestimate:
        u_hat = np.zeros_like(Y)
        v_hat = np.zeros_like(D)
        for idx, (train_index, test_index) in enumerate(smpls):
            v_hat[test_index] = D[test_index] - m_hat[idx]
            u_hat[test_index] = Y[test_index] - g_hat[idx]
        se = np.sqrt(var_plr(theta_hat, D, u_hat, v_hat, score, n_obs))
    else:
        ses = np.zeros(len(smpls))
        for idx, (train_index, test_index) in enumerate(smpls):
            v_hat = D[test_index] - m_hat[idx]
            u_hat = Y[test_index] - g_hat[idx]
            ses[idx] = var_plr(theta_hat, D[test_index],
                               u_hat, v_hat,
                               score, n_obs)
        se = np.sqrt(np.mean(ses))
    
    return theta_hat, se

def plr_dml2(Y, X, D, g_hat, m_hat, smpls, score):
    thetas = np.zeros(len(smpls))
    n_obs = len(Y)
    u_hat = np.zeros_like(Y)
    v_hat = np.zeros_like(D)
    for idx, (train_index, test_index) in enumerate(smpls):
        v_hat[test_index] = D[test_index] - m_hat[idx]
        u_hat[test_index] = Y[test_index] - g_hat[idx]
    theta_hat = plr_orth(v_hat, u_hat, D, score)
    se = np.sqrt(var_plr(theta_hat, D, u_hat, v_hat, score, n_obs))
    
    return theta_hat, se
    
def var_plr(theta, d, u_hat, v_hat, score, n_obs):
    if score == 'partialling out':
        var = 1/n_obs * 1/np.power(np.mean(np.multiply(v_hat, v_hat)), 2) * \
              np.mean(np.power(np.multiply(u_hat - v_hat*theta, v_hat), 2))
    elif score == 'IV-type':
        var = 1/n_obs * 1/np.power(np.mean(np.multiply(v_hat, d)), 2) * \
              np.mean(np.power(np.multiply(u_hat - d*theta, v_hat), 2))
    else:
        raise ValueError('invalid score')
    
    return var

def plr_orth(v_hat, u_hat, D, score):
    if score == 'IV-type':
        res = np.mean(np.multiply(v_hat, u_hat))/np.mean(np.multiply(v_hat, D))
    elif score == 'partialling out':
        res = scipy.linalg.lstsq(v_hat.reshape(-1, 1), u_hat)[0]
    
    return res

def boot_plr(theta, Y, D, g_hat, m_hat, smpls, score, se, bootstrap, n_rep, dml_procedure):
    u_hat = np.zeros_like(Y)
    v_hat = np.zeros_like(D)
    n_folds = len(smpls)
    J = np.zeros(n_folds)
    for idx, (train_index, test_index) in enumerate(smpls):
        v_hat[test_index] = D[test_index] - m_hat[idx]
        u_hat[test_index] = Y[test_index] - g_hat[idx]
        if dml_procedure == 'dml1':
            if score == 'partialling out':
                J[idx] = np.mean(-np.multiply(v_hat[test_index], v_hat[test_index]))
            elif score == 'IV-type':
                J[idx] = np.mean(-np.multiply(v_hat[test_index], D[test_index]))

    if dml_procedure == 'dml2':
        if score == 'partialling out':
            J = np.mean(-np.multiply(v_hat, v_hat))
        elif score == 'IV-type':
            J = np.mean(-np.multiply(v_hat, D))

    if score == 'partialling out':
        score = np.multiply(u_hat - v_hat * theta, v_hat)
    elif score == 'IV-type':
        score = np.multiply(u_hat - D * theta, v_hat)
    else:
        raise ValueError('invalid score')

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
            xx = xx_sample[i_rep, :]
            yy = yy_sample[i_rep, :]
            weights = xx / np.sqrt(2) + (np.power(yy, 2) - 1)/2
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
