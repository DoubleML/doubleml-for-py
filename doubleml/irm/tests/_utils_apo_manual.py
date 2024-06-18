import numpy as np
from sklearn.base import clone, is_classifier

from ...tests._utils_boot import boot_manual, draw_weights
from ...tests._utils import fit_predict, fit_predict_proba, tune_grid_search

from ...utils._estimation import _normalize_ipw
from ...utils._checks import _check_is_propensity


def fit_apo(y, x, d,
            learner_g, learner_m, treatment_level, all_smpls, score,
            n_rep=1, g0_params=None, g1_params=None, m_params=None,
            normalize_ipw=False, trimming_threshold=1e-2):
    n_obs = len(y)
    treated = (d == treatment_level)

    thetas = np.zeros(n_rep)
    ses = np.zeros(n_rep)
    all_g_hat0 = list()
    all_g_hat1 = list()
    all_m_hat = list()

    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]
        g_hat0, g_hat1, m_hat = fit_nuisance_apo(y, x, d, treated,
                                                 learner_g, learner_m, smpls, score,
                                                 g0_params=g0_params, g1_params=g1_params, m_params=m_params,
                                                 trimming_threshold=trimming_threshold)

        all_g_hat0.append(g_hat0)
        all_g_hat1.append(g_hat1)
        all_m_hat.append(m_hat)

        thetas[i_rep], ses[i_rep] = apo_dml2(y, x, d, treated,
                                             g_hat0, g_hat1, m_hat,
                                             smpls, score, normalize_ipw)

    theta = np.median(thetas)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(thetas - theta, 2)) / n_obs)

    res = {'theta': theta, 'se': se,
           'thetas': thetas, 'ses': ses,
           'all_g_hat0': all_g_hat0, 'all_g_hat1': all_g_hat1, 'all_m_hat': all_m_hat}

    return res


def fit_nuisance_apo(y, x, d, treated,
                     learner_g, learner_m, smpls, score,
                     g0_params=None, g1_params=None, m_params=None,
                     trimming_threshold=1e-12):
    ml_g0 = clone(learner_g)
    ml_g1 = clone(learner_g)

    train_cond0 = np.where(treated == 0)[0]
    if is_classifier(learner_g):
        g_hat0_list = fit_predict_proba(y, x, ml_g0, g0_params, smpls,
                                        train_cond=train_cond0)
    else:
        g_hat0_list = fit_predict(y, x, ml_g0, g0_params, smpls,
                                  train_cond=train_cond0)

    train_cond1 = np.where(treated == 1)[0]
    if is_classifier(learner_g):
        g_hat1_list = fit_predict_proba(y, x, ml_g1, g1_params, smpls,
                                        train_cond=train_cond1)
    else:
        g_hat1_list = fit_predict(y, x, ml_g1, g1_params, smpls,
                                  train_cond=train_cond1)

    ml_m = clone(learner_m)
    m_hat_list = fit_predict_proba(treated, x, ml_m, m_params, smpls,
                                   trimming_threshold=trimming_threshold)

    return g_hat0_list, g_hat1_list, m_hat_list


def compute_residuals(y, g_hat0_list, g_hat1_list, m_hat_list, smpls):
    u_hat0 = np.full_like(y, np.nan, dtype='float64')
    u_hat1 = np.full_like(y, np.nan, dtype='float64')
    g_hat0 = np.full_like(y, np.nan, dtype='float64')
    g_hat1 = np.full_like(y, np.nan, dtype='float64')
    m_hat = np.full_like(y, np.nan, dtype='float64')
    for idx, (_, test_index) in enumerate(smpls):
        u_hat0[test_index] = y[test_index] - g_hat0_list[idx]
        u_hat1[test_index] = y[test_index] - g_hat1_list[idx]
        g_hat0[test_index] = g_hat0_list[idx]
        g_hat1[test_index] = g_hat1_list[idx]
        m_hat[test_index] = m_hat_list[idx]

    _check_is_propensity(m_hat, 'learner_m', 'ml_m', smpls, eps=1e-12)
    return u_hat0, u_hat1, g_hat0, g_hat1, m_hat


def apo_dml2(y, x, d, treated, g_hat0_list, g_hat1_list, m_hat_list, smpls, score, normalize_ipw):
    n_obs = len(y)
    u_hat0, u_hat1, g_hat0, g_hat1, m_hat = compute_residuals(
        y, g_hat0_list, g_hat1_list, m_hat_list, smpls
    )

    if normalize_ipw:
        m_hat_adj = _normalize_ipw(m_hat, treated)
    else:
        m_hat_adj = m_hat

    theta_hat = apo_orth(g_hat0, g_hat1, m_hat_adj,
                         u_hat0, u_hat1, treated, score)

    se = np.sqrt(var_apo(theta_hat, g_hat0, g_hat1,
                         m_hat_adj,
                         u_hat0, u_hat1,
                         treated, score, n_obs))

    return theta_hat, se


def apo_orth(g_hat0, g_hat1, m_hat, u_hat0, u_hat1, treated, score):
    res = np.mean(g_hat1 + np.divide(np.multiply(treated, u_hat1), m_hat))
    return res


def var_apo(theta, g_hat0, g_hat1, m_hat, u_hat0, u_hat1, treated, score, n_obs):
    var = 1/n_obs * np.mean(np.power(g_hat1 + np.divide(np.multiply(treated, u_hat1), m_hat), 2))
    return var
