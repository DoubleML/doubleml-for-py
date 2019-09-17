import numpy as np
import pytest
import math
import scipy


def fit_nuisance_irm(Y, X, D, ml_m, ml_g, smpls, inf_model):
    g_hat0 = []
    g_hat1 = []
    for idx, (train_index, test_index) in enumerate(smpls):
        train_index0 =np.intersect1d(np.where(D==0)[0], train_index)
        g_hat0.append(ml_g.fit(X[train_index0],Y[train_index0]).predict(X[test_index]))
    
    if inf_model == 'ATE':
        for idx, (train_index, test_index) in enumerate(smpls):
            train_index1 =np.intersect1d(np.where(D==1)[0], train_index)
            g_hat1.append(ml_g.fit(X[train_index1],Y[train_index1]).predict(X[test_index]))
    else:
        for idx, (train_index, test_index) in enumerate(smpls):
            # fill it up, but its not further used
            g_hat1.append(np.zeros_like(g_hat0[idx]))
    
    m_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        m_hat.append(ml_m.fit(X[train_index],D[train_index]).predict_proba(X[test_index])[:, 1])
    
    return g_hat0, g_hat1, m_hat

def irm_dml1(Y, X, D, g_hat0, g_hat1, m_hat, smpls, inf_model):
    thetas = np.zeros(len(smpls))
    
    for idx, (train_index, test_index) in enumerate(smpls):
        u_hat0 = Y[test_index] - g_hat0[idx]
        u_hat1 = Y[test_index] - g_hat1[idx]
        thetas[idx] = irm_orth(g_hat0[idx], g_hat1[idx],
                               m_hat[idx], u_hat0, u_hat1,
                               D[test_index], inf_model)
    theta_hat = np.mean(thetas)
    
    ses = np.zeros(len(smpls))
    for idx, (train_index, test_index) in enumerate(smpls):
        u_hat0 = Y[test_index] - g_hat0[idx]
        u_hat1 = Y[test_index] - g_hat1[idx]
        ses[idx] = var_irm(theta_hat, g_hat0[idx], g_hat1[idx],
                           m_hat[idx], u_hat0, u_hat1,
                           D[test_index], inf_model)
    se = np.sqrt(np.mean(ses))
    
    return theta_hat, se

def irm_dml2(Y, X, D, g_hat0, g_hat1, m_hat, smpls, inf_model):
    u_hat0 = np.zeros_like(Y)
    u_hat1 = np.zeros_like(Y)
    g_hat0_all = np.zeros_like(Y)
    g_hat1_all = np.zeros_like(Y)
    m_hat_all = np.zeros_like(D)
    for idx, (train_index, test_index) in enumerate(smpls):
        u_hat0[test_index] = Y[test_index] - g_hat0[idx]
        u_hat1[test_index] = Y[test_index] - g_hat1[idx]
        g_hat0_all[test_index] = g_hat0[idx]
        g_hat1_all[test_index] = g_hat1[idx]
        m_hat_all[test_index] = m_hat[idx]
    theta_hat = irm_orth(g_hat0_all, g_hat1_all, m_hat_all, u_hat0, u_hat1, D, inf_model)
    se = np.sqrt(var_irm(theta_hat, g_hat0_all, g_hat1_all,
                         m_hat_all, u_hat0, u_hat1,
                         D, inf_model))
    
    return theta_hat, se
    
def var_irm(theta, g_hat0, g_hat1, m_hat, u_hat0, u_hat1, D, inf_model):
    n_obs = len(D)
    
    #if se_type == 'ATE':
    #    var = 1/n_obs * 1/np.power(np.mean(np.multiply(v_hat, v_hat)), 2) * \
    #          np.mean(np.power(np.multiply(u_hat - v_hat*theta, v_hat), 2))
    #elif se_type == 'ATTE':
    #    var = 1/n_obs * 1/np.power(np.mean(np.multiply(v_hat, d)), 2) * \
    #          np.mean(np.power(np.multiply(u_hat - d*theta, v_hat), 2))
    #else:
    #    raise ValueError('invalid se_type')
    var = 1.0
    
    return var

def irm_orth(g_hat0, g_hat1, m_hat, u_hat0, u_hat1, D, inf_model):
    if inf_model == 'ATE':
        res = np.mean(g_hat1 - g_hat0 \
                      + np.divide(np.multiply(D, u_hat1), m_hat) \
                      - np.divide(np.multiply(1.-D, u_hat0), 1.-m_hat))
    elif inf_model == 'ATTE':
        Ep = np.mean(D)
        
        res = np.mean(np.multiply(D, u_hat0)/Ep \
                      - np.divide(np.multiply(m_hat, np.multiply(1.-D, u_hat0)), Ep*(1.-m_hat))) \
              / np.mean(D/Ep)
    
    return res

def boot_irm(theta, Y, D, g_hat, m_hat, smpls, inf_model, se, bootstrap, n_rep):
    u_hat = np.zeros_like(Y)
    v_hat = np.zeros_like(D)
    for idx, (train_index, test_index) in enumerate(smpls):
        v_hat[test_index] = D[test_index] - m_hat[idx]
        u_hat[test_index] = Y[test_index] - g_hat[idx]
    
    if inf_model == 'DML2018':
        score = np.multiply(u_hat - v_hat*theta, v_hat)
        J = np.mean(-np.multiply(v_hat, v_hat))
    elif inf_model == 'IV-type':
        score = np.multiply(u_hat - D*theta, v_hat)
        J = np.mean(-np.multiply(v_hat, D))
    else:
        raise ValueError('invalid se_type')
    
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
        
        boot_theta[i_rep] = np.mean(np.multiply(np.divide(weights, se),
                                               score / J))
    
    return boot_theta
