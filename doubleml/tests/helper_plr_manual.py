import numpy as np
import scipy
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.base import clone

from doubleml.tests.helper_boot import boot_manual

def fit_nuisance_plr(Y, X, D, learner_m, learner_g, smpls, g_params=None, m_params=None):
    ml_g = clone(learner_g)
    g_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if g_params is not None:
            ml_g.set_params(**g_params[idx])
        g_hat.append(ml_g.fit(X[train_index], Y[train_index]).predict(X[test_index]))

    ml_m = clone(learner_m)
    m_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if m_params is not None:
            ml_m.set_params(**m_params[idx])
        m_hat.append(ml_m.fit(X[train_index], D[train_index]).predict(X[test_index]))
    
    return g_hat, m_hat


def tune_nuisance_plr(Y, X, D, ml_m, ml_g, smpls, n_folds_tune, param_grid_g, param_grid_m):
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
        m_tune_res[idx] = m_grid_search.fit(X[train_index, :], D[train_index])

    g_best_params = [xx.best_params_ for xx in g_tune_res]
    m_best_params = [xx.best_params_ for xx in m_tune_res]

    return g_best_params, m_best_params


def plr_dml1(Y, X, D, g_hat, m_hat, smpls, score):
    thetas = np.zeros(len(smpls))
    n_obs = len(Y)
    
    for idx, (train_index, test_index) in enumerate(smpls):
        v_hat = D[test_index] - m_hat[idx]
        u_hat = Y[test_index] - g_hat[idx]
        thetas[idx] = plr_orth(v_hat, u_hat, D[test_index], score)
    theta_hat = np.mean(thetas)

    if len(smpls) > 1:
        u_hat = np.zeros_like(Y)
        v_hat = np.zeros_like(D)
        for idx, (train_index, test_index) in enumerate(smpls):
            v_hat[test_index] = D[test_index] - m_hat[idx]
            u_hat[test_index] = Y[test_index] - g_hat[idx]
        se = np.sqrt(var_plr(theta_hat, D, u_hat, v_hat, score, n_obs))
    else:
        assert len(smpls) == 1
        test_index = smpls[0][1]
        n_obs = len(test_index)
        v_hat = D[test_index] - m_hat[0]
        u_hat = Y[test_index] - g_hat[0]
        se = np.sqrt(var_plr(theta_hat, D[test_index], u_hat, v_hat, score, n_obs))
    
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

def boot_plr(theta, Y, D, g_hat, m_hat, smpls, score, se, bootstrap, n_rep, dml_procedure, apply_cross_fitting=True):
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
        psi = np.multiply(u_hat - v_hat * theta, v_hat)
    elif score == 'IV-type':
        psi = np.multiply(u_hat - D * theta, v_hat)
    else:
        raise ValueError('invalid score')

    boot_theta, boot_t_stat = boot_manual(psi, J, smpls, se, bootstrap, n_rep, dml_procedure, apply_cross_fitting)
    
    return boot_theta, boot_t_stat
