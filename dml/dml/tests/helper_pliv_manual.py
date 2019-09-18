import numpy as np
import pytest
import math
import scipy


def fit_nuisance_pliv(Y, X, D, Z, ml_m, ml_g, ml_r, smpls):
    g_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        g_hat.append(ml_g.fit(X[train_index],Y[train_index]).predict(X[test_index]))
    
    m_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        m_hat.append(ml_m.fit(X[train_index],Z[train_index]).predict(X[test_index]))
    
    r_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        r_hat.append(ml_r.fit(X[train_index],D[train_index]).predict(X[test_index]))
    
    return g_hat, m_hat, r_hat

def pliv_dml1(Y, X, D, Z, g_hat, m_hat, r_hat, smpls, inf_model):
    thetas = np.zeros(len(smpls))
    
    for idx, (train_index, test_index) in enumerate(smpls):
        u_hat = Y[test_index] - g_hat[idx]
        v_hat = Z[test_index] - m_hat[idx]
        w_hat = D[test_index] - r_hat[idx]
        thetas[idx] = pliv_orth(u_hat, v_hat, w_hat, D[test_index], inf_model)
    theta_hat = np.mean(thetas)
    
    ses = np.zeros(len(smpls))
    for idx, (train_index, test_index) in enumerate(smpls):
        u_hat = Y[test_index] - g_hat[idx]
        v_hat = Z[test_index] - m_hat[idx]
        w_hat = D[test_index] - r_hat[idx]
        ses[idx] = var_pliv(theta_hat, D[test_index],
                            u_hat, v_hat, w_hat,
                            inf_model)
    se = np.sqrt(np.mean(ses))
    
    return theta_hat, se

def pliv_dml2(Y, X, D, Z, g_hat, m_hat, r_hat, smpls, inf_model):
    thetas = np.zeros(len(smpls))
    u_hat = np.zeros_like(Y)
    v_hat = np.zeros_like(Z)
    w_hat = np.zeros_like(D)
    for idx, (train_index, test_index) in enumerate(smpls):
        u_hat[test_index] = Y[test_index] - g_hat[idx]
        v_hat[test_index] = Z[test_index] - m_hat[idx]
        w_hat[test_index] = D[test_index] - r_hat[idx]
    theta_hat = pliv_orth(u_hat, v_hat, w_hat, D, inf_model)
    se = np.sqrt(var_pliv(theta_hat, D, u_hat, v_hat, w_hat, inf_model))
    
    return theta_hat, se
    
def var_pliv(theta, d, u_hat, v_hat, w_hat, se_type):
    n_obs = len(u_hat)
    
    if se_type == 'DML2018':
        var = 1/n_obs * 1/np.power(np.mean(np.multiply(v_hat, w_hat)), 2) * \
              np.mean(np.power(np.multiply(u_hat - w_hat*theta, v_hat), 2))
    else:
        raise ValueError('invalid se_type')
    
    return var

def pliv_orth(u_hat, v_hat, w_hat, D, inf_model):
    if inf_model == 'DML2018':
        res = np.mean(np.multiply(v_hat, u_hat))/np.mean(np.multiply(v_hat, w_hat))
    else:
      raise ValueError('invalid inf_model')
    
    return res

def boot_pliv(theta, Y, D, Z, g_hat, m_hat, r_hat, smpls, inf_model, se, bootstrap, n_rep):
    u_hat = np.zeros_like(Y)
    v_hat = np.zeros_like(Z)
    w_hat = np.zeros_like(D)
    for idx, (train_index, test_index) in enumerate(smpls):
        u_hat[test_index] = Y[test_index] - g_hat[idx]
        v_hat[test_index] = Z[test_index] - m_hat[idx]
        w_hat[test_index] = D[test_index] - r_hat[idx]
    
    if inf_model == 'DML2018':
        score = np.multiply(u_hat - w_hat*theta, v_hat)
        J = np.mean(-np.multiply(v_hat, w_hat))
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
