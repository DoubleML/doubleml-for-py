import numpy as np
import scipy
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.base import clone

from ._utils_boot import boot_manual, draw_weights


def fit_nuisance_plr(y, x, d, learner_m, learner_g, smpls, g_params=None, m_params=None):
    ml_g = clone(learner_g)
    g_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if g_params is not None:
            ml_g.set_params(**g_params[idx])
        g_hat.append(ml_g.fit(x[train_index], y[train_index]).predict(x[test_index]))

    ml_m = clone(learner_m)
    m_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if m_params is not None:
            ml_m.set_params(**m_params[idx])
        m_hat.append(ml_m.fit(x[train_index], d[train_index]).predict(x[test_index]))

    return g_hat, m_hat


def fit_nuisance_plr_classifier(y, x, d, learner_m, learner_g, smpls, g_params=None, m_params=None):
    ml_g = clone(learner_g)
    g_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if g_params is not None:
            ml_g.set_params(**g_params[idx])
        g_hat.append(ml_g.fit(x[train_index], y[train_index]).predict(x[test_index]))

    ml_m = clone(learner_m)
    m_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if m_params is not None:
            ml_m.set_params(**m_params[idx])
        m_hat.append(ml_m.fit(x[train_index], d[train_index]).predict_proba(x[test_index])[:, 1])

    return g_hat, m_hat


def tune_nuisance_plr(y, x, d, ml_m, ml_g, smpls, n_folds_tune, param_grid_g, param_grid_m):
    g_tune_res = [None] * len(smpls)
    for idx, (train_index, _) in enumerate(smpls):
        g_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
        g_grid_search = GridSearchCV(ml_g, param_grid_g,
                                     cv=g_tune_resampling)
        g_tune_res[idx] = g_grid_search.fit(x[train_index, :], y[train_index])

    m_tune_res = [None] * len(smpls)
    for idx, (train_index, _) in enumerate(smpls):
        m_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
        m_grid_search = GridSearchCV(ml_m, param_grid_m,
                                     cv=m_tune_resampling)
        m_tune_res[idx] = m_grid_search.fit(x[train_index, :], d[train_index])

    g_best_params = [xx.best_params_ for xx in g_tune_res]
    m_best_params = [xx.best_params_ for xx in m_tune_res]

    return g_best_params, m_best_params


def plr_dml1(y, x, d, g_hat, m_hat, smpls, score):
    thetas = np.zeros(len(smpls))
    n_obs = len(y)

    for idx, (_, test_index) in enumerate(smpls):
        v_hat = d[test_index] - m_hat[idx]
        u_hat = y[test_index] - g_hat[idx]
        thetas[idx] = plr_orth(v_hat, u_hat, d[test_index], score)
    theta_hat = np.mean(thetas)

    if len(smpls) > 1:
        u_hat = np.zeros_like(y, dtype='float64')
        v_hat = np.zeros_like(d, dtype='float64')
        for idx, (_, test_index) in enumerate(smpls):
            v_hat[test_index] = d[test_index] - m_hat[idx]
            u_hat[test_index] = y[test_index] - g_hat[idx]
        se = np.sqrt(var_plr(theta_hat, d, u_hat, v_hat, score, n_obs))
    else:
        assert len(smpls) == 1
        test_index = smpls[0][1]
        n_obs = len(test_index)
        v_hat = d[test_index] - m_hat[0]
        u_hat = y[test_index] - g_hat[0]
        se = np.sqrt(var_plr(theta_hat, d[test_index], u_hat, v_hat, score, n_obs))

    return theta_hat, se


def plr_dml2(y, x, d, g_hat, m_hat, smpls, score):
    n_obs = len(y)
    u_hat = np.zeros_like(y, dtype='float64')
    v_hat = np.zeros_like(d, dtype='float64')
    for idx, (_, test_index) in enumerate(smpls):
        v_hat[test_index] = d[test_index] - m_hat[idx]
        u_hat[test_index] = y[test_index] - g_hat[idx]
    theta_hat = plr_orth(v_hat, u_hat, d, score)
    se = np.sqrt(var_plr(theta_hat, d, u_hat, v_hat, score, n_obs))

    return theta_hat, se


def var_plr(theta, d, u_hat, v_hat, score, n_obs):
    if score == 'partialling out':
        var = 1/n_obs * 1/np.power(np.mean(np.multiply(v_hat, v_hat)), 2) * \
            np.mean(np.power(np.multiply(u_hat - v_hat*theta, v_hat), 2))
    else:
        assert score == 'IV-type'
        var = 1/n_obs * 1/np.power(np.mean(np.multiply(v_hat, d)), 2) * \
            np.mean(np.power(np.multiply(u_hat - d*theta, v_hat), 2))

    return var


def plr_orth(v_hat, u_hat, d, score):
    if score == 'IV-type':
        res = np.mean(np.multiply(v_hat, u_hat))/np.mean(np.multiply(v_hat, d))
    else:
        assert score == 'partialling out'
        res = scipy.linalg.lstsq(v_hat.reshape(-1, 1), u_hat)[0]

    return res


def boot_plr(theta, y, d, g_hat, m_hat, smpls, score, se, bootstrap, n_rep, dml_procedure, apply_cross_fitting=True):
    if apply_cross_fitting:
        n_obs = len(y)
    else:
        test_index = smpls[0][1]
        n_obs = len(test_index)
    weights = draw_weights(bootstrap, n_rep, n_obs)

    if np.isscalar(theta):
        n_d = 1
    else:
        n_d = len(theta)
    if n_d > 1:
        boot_theta = np.full((n_d, n_rep), np.nan)
        boot_t_stat = np.full((n_d, n_rep), np.nan)
        for i_d in range(n_d):
            boot_theta[i_d, :], boot_t_stat[i_d, :] = boot_plr_single_treat(theta[i_d],
                                                                            y, d[:, i_d],
                                                                            g_hat[i_d], m_hat[i_d],
                                                                            smpls, score,
                                                                            se[i_d],
                                                                            weights, n_rep,
                                                                            dml_procedure,
                                                                            apply_cross_fitting)
    else:
        boot_theta, boot_t_stat = boot_plr_single_treat(theta,
                                                        y, d,
                                                        g_hat, m_hat,
                                                        smpls, score,
                                                        se,
                                                        weights, n_rep,
                                                        dml_procedure,
                                                        apply_cross_fitting)

    return boot_theta, boot_t_stat


def boot_plr_single_treat(theta, y, d, g_hat, m_hat, smpls, score, se, weights, n_rep, dml_procedure, apply_cross_fitting):
    u_hat = np.zeros_like(y, dtype='float64')
    v_hat = np.zeros_like(d, dtype='float64')
    n_folds = len(smpls)
    J = np.zeros(n_folds)
    for idx, (_, test_index) in enumerate(smpls):
        v_hat[test_index] = d[test_index] - m_hat[idx]
        u_hat[test_index] = y[test_index] - g_hat[idx]
        if dml_procedure == 'dml1':
            if score == 'partialling out':
                J[idx] = np.mean(-np.multiply(v_hat[test_index], v_hat[test_index]))
            else:
                assert score == 'IV-type'
                J[idx] = np.mean(-np.multiply(v_hat[test_index], d[test_index]))

    if dml_procedure == 'dml2':
        if score == 'partialling out':
            J = np.mean(-np.multiply(v_hat, v_hat))
        else:
            assert score == 'IV-type'
            J = np.mean(-np.multiply(v_hat, d))

    if score == 'partialling out':
        psi = np.multiply(u_hat - v_hat * theta, v_hat)
    else:
        assert score == 'IV-type'
        psi = np.multiply(u_hat - d * theta, v_hat)

    boot_theta, boot_t_stat = boot_manual(psi, J, smpls, se, weights, n_rep, dml_procedure, apply_cross_fitting)

    return boot_theta, boot_t_stat
