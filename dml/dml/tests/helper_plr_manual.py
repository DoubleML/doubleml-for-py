import numpy as np
import pytest
import math
import scipy

from dml.double_ml_plr import DoubleMLPLR


def fit_nuisance_plr(Y, X, D, ml_m, ml_g, smpls):
    g_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        g_hat.append(ml_g.fit(X[train_index],Y[train_index]).predict(X[test_index]))
    
    m_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        m_hat.append(ml_m.fit(X[train_index],D[train_index]).predict(X[test_index]))
    
    return g_hat, m_hat

def plr_dml1(Y, X, D, g_hat, m_hat, smpls, inf_model):
    thetas = np.zeros(len(smpls))
    
    for idx, (train_index, test_index) in enumerate(smpls):
        v_hat = D[test_index] - m_hat[idx]
        u_hat = Y[test_index] - g_hat[idx]
        thetas[idx] = plr_orth(v_hat, u_hat, D[test_index], inf_model)
    theta_hat = np.mean(thetas)
    
    ses = np.zeros(len(smpls))
    for idx, (train_index, test_index) in enumerate(smpls):
        v_hat = D[test_index] - m_hat[idx]
        u_hat = Y[test_index] - g_hat[idx]
        ses[idx] = var_plr(theta_hat, D[test_index],
                           u_hat, v_hat,
                           inf_model)
    se = np.sqrt(np.mean(ses))
    
    return theta_hat, se

def plr_dml2(Y, X, D, g_hat, m_hat, smpls, inf_model):
    thetas = np.zeros(len(smpls))
    u_hat = np.zeros_like(Y)
    v_hat = np.zeros_like(D)
    for idx, (train_index, test_index) in enumerate(smpls):
        v_hat[test_index] = D[test_index] - m_hat[idx]
        u_hat[test_index] = Y[test_index] - g_hat[idx]
    theta_hat = plr_orth(v_hat, u_hat, D, inf_model)
    se = np.sqrt(var_plr(theta_hat, D, u_hat, v_hat, inf_model))
    
    return theta_hat, se
    
def var_plr(theta, d, u_hat, v_hat, se_type):
    n_obs = len(u_hat)
    
    if se_type == 'DML2018':
        var = 1/n_obs * 1/np.power(np.mean(np.multiply(v_hat, v_hat)), 2) * \
              np.mean(np.power(np.multiply(u_hat - v_hat*theta, v_hat), 2))
    elif se_type == 'IV-type':
        var = 1/n_obs * 1/np.power(np.mean(np.multiply(v_hat, d)), 2) * \
              np.mean(np.power(np.multiply(u_hat - d*theta, v_hat), 2))
    else:
        raise ValueError('invalid se_type')
    
    return var

def plr_orth(v_hat, u_hat, D, inf_model):
    if inf_model == 'IV-type':
        res = np.mean(np.multiply(v_hat, u_hat))/np.mean(np.multiply(v_hat, D))
    elif inf_model == 'DML2018':
        res = scipy.linalg.lstsq(v_hat.reshape(-1, 1), u_hat)[0]
    
    return res

def boot_plr(theta, Y, D, g_hat, m_hat, smpls, inf_model, se, bootstrap, n_rep):
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
