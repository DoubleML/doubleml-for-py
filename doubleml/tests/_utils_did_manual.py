import numpy as np
from sklearn.base import clone, is_classifier

from ._utils_boot import boot_manual, draw_weights
from ._utils import fit_predict, fit_predict_proba, tune_grid_search

from .._utils import _check_is_propensity, _normalize_ipw


def fit_did(y, x, d,
            learner_g, learner_m, all_smpls, dml_procedure, score,
            n_rep=1, g0_params=None, g1_params=None, m_params=None,
            trimming_threshold=1e-2):
    n_obs = len(y)

    thetas = np.zeros(n_rep)
    ses = np.zeros(n_rep)
    all_g_hat0 = list()
    all_g_hat1 = list()
    all_m_hat = list()
    all_p_hat = list()
    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]

        g_hat0, g_hat1, m_hat, p_hat = fit_nuisance_did(y, x, d,
                                                        learner_g, learner_m, smpls,
                                                        score,
                                                        g0_params=g0_params, g1_params=g1_params, m_params=m_params,
                                                        trimming_threshold=trimming_threshold)

        all_g_hat0.append(g_hat0)
        all_g_hat1.append(g_hat1)
        all_m_hat.append(m_hat)
        all_p_hat.append(p_hat)

        if dml_procedure == 'dml1':
            thetas[i_rep], ses[i_rep] = did_dml1(y, x, d,
                                                 g_hat0, g_hat1, m_hat, p_hat,
                                                 smpls, score)
        else:
            assert dml_procedure == 'dml2'
            thetas[i_rep], ses[i_rep] = did_dml2(y, x, d,
                                                 g_hat0, g_hat1, m_hat, p_hat,
                                                 smpls, score)

    theta = np.median(thetas)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(thetas - theta, 2)) / n_obs)

    res = {'theta': theta, 'se': se,
           'thetas': thetas, 'ses': ses,
           'all_g_hat0': all_g_hat0, 'all_g_hat1': all_g_hat1, 'all_m_hat': all_m_hat, 'all_p_hat': all_p_hat}

    return res


def fit_nuisance_did(y, x, d, learner_g, learner_m, smpls, score,
                     g0_params=None, g1_params=None, m_params=None,
                     trimming_threshold=1e-12):
    ml_g0 = clone(learner_g)
    ml_g1 = clone(learner_g)
    train_cond0 = np.where(d == 0)[0]
    g_hat0_list = fit_predict(y, x, ml_g0, g0_params, smpls,
                                train_cond=train_cond0)

    if score == 'PA-2':
        train_cond1 = np.where(d == 1)[0]
        g_hat1_list = fit_predict(y, x, ml_g1, g1_params, smpls,
                                    train_cond=train_cond1)

    else:
        assert (score == 'PA-1') | (score == 'DR')
        g_hat1_list = list()
        for idx, _ in enumerate(smpls):
            # fill it up, but its not further used
            g_hat1_list.append(np.zeros_like(g_hat0_list[idx], dtype='float64'))

    ml_m = clone(learner_m)
    m_hat_list = fit_predict_proba(d, x, ml_m, m_params, smpls,
                                   trimming_threshold=trimming_threshold)

    p_hat_list = []
    for (train_index, test_index) in smpls:
        p_hat_list.append(np.mean(d[train_index]))

    return g_hat0_list, g_hat1_list, m_hat_list, p_hat_list


def compute_did_residuals(y, g_hat0_list, g_hat1_list, m_hat_list, p_hat_list, smpls):
    y_resid_d0 = np.full_like(y, np.nan, dtype='float64')
    g_hat0 = np.full_like(y, np.nan, dtype='float64')
    g_hat1 = np.full_like(y, np.nan, dtype='float64')
    m_hat = np.full_like(y, np.nan, dtype='float64')
    p_hat = np.full_like(y, np.nan, dtype='float64')
    for idx, (_, test_index) in enumerate(smpls):
        y_resid_d0[test_index] = y[test_index] - g_hat0_list[idx]
        g_hat0[test_index] = g_hat0_list[idx]
        g_hat1[test_index] = g_hat1_list[idx]
        m_hat[test_index] = m_hat_list[idx]
        p_hat[test_index] = p_hat_list[idx]

    _check_is_propensity(m_hat, 'learner_m', 'ml_m', smpls, eps=1e-12)
    return y_resid_d0, g_hat0, g_hat1, m_hat, p_hat


def did_dml1(y, x, d, g_hat0_list, g_hat1_list, m_hat_list, p_hat_list, smpls, score):
    thetas = np.zeros(len(smpls))
    n_obs = len(y)
    y_resid_d0, g_hat0, g_hat1, m_hat, p_hat = compute_did_residuals(
        y, g_hat0_list, g_hat1_list, m_hat_list, p_hat_list, smpls)
    
    psi_a, psi_b = did_score_elements(g_hat0, g_hat1, m_hat, p_hat,
                                      y_resid_d0, d, score)
    for idx, (_, test_index) in enumerate(smpls):
        thetas[idx] = - np.mean(psi_b[test_index]) / np.mean(psi_a[test_index])
    theta_hat = np.mean(thetas)

    if len(smpls) > 1:
        se = np.sqrt(var_did(theta_hat, g_hat0, g_hat1,
                             m_hat, p_hat,
                             y_resid_d0,
                             d, score, n_obs))
    else:
        assert len(smpls) == 1
        test_index = smpls[0][1]
        n_obs = len(test_index)
        se = np.sqrt(var_did(theta_hat, g_hat0[test_index], g_hat1[test_index],
                             m_hat[test_index], p_hat[test_index],
                             y_resid_d0[test_index],
                             d[test_index], score, n_obs))

    return theta_hat, se


def did_dml2(y, x, d, g_hat0_list, g_hat1_list, m_hat_list, p_hat_list, smpls, score):
    n_obs = len(y)
    y_resid_d0, g_hat0, g_hat1, m_hat, p_hat = compute_did_residuals(
        y, g_hat0_list, g_hat1_list, m_hat_list, p_hat_list, smpls)

    psi_a, psi_b = did_score_elements(g_hat0, g_hat1, m_hat, p_hat,
                                      y_resid_d0, d, score)
    
    theta_hat = - np.mean(psi_b) / np.mean(psi_a)
    se = np.sqrt(var_did(theta_hat, g_hat0, g_hat1,
                         m_hat, p_hat,
                         y_resid_d0,
                         d, score, n_obs))

    return theta_hat, se


def did_score_elements(g_hat0, g_hat1, m_hat, p_hat, y_resid_d0, d, score):

    if score == 'PA-1':
        psi_a = -1.0 * np.divide(d, p_hat)
        y_resid_d0_weight = np.divide(d-m_hat, np.multiply(p_hat, 1.0-m_hat))
        psi_b = np.multiply(y_resid_d0_weight, y_resid_d0)
        
    elif score == 'PA-2':
        psi_a = -1.0 * np.ones_like(d)
        y_resid_d0_weight = np.divide(d-m_hat, np.multiply(p_hat, 1.0-m_hat))
        psi_b_1 = np.multiply(y_resid_d0_weight, y_resid_d0)
        psi_b_2 = np.multiply(1.0-np.divide(d, p_hat), g_hat1 - g_hat0)
        psi_b = psi_b_1 + psi_b_2
    
    else:
        assert score == 'DR'
        psi_a = -1.0 * np.divide(d, np.mean(d))
        propensity_weight = np.divide(m_hat, 1.0-m_hat)
        y_resid_d0_weight = np.divide(d, np.mean(d)) \
            - np.divide(np.multiply(1.0-d, propensity_weight), np.mean(np.multiply(1.0-d, propensity_weight)))
        psi_b = np.multiply(y_resid_d0_weight, y_resid_d0)

    return psi_a, psi_b


def var_did(theta, g_hat0, g_hat1, m_hat, p_hat, y_resid_d0, d, score, n_obs):

    if score == 'PA-1':
        psi_a = -1.0 * np.divide(d, p_hat)
        y_resid_d0_weight = np.divide(d-m_hat, np.multiply(p_hat, 1.0-m_hat))
        psi_b = np.multiply(y_resid_d0_weight, y_resid_d0)

    elif score == 'PA-2':
        psi_a = -1.0 * np.ones_like(d)
        y_resid_d0_weight = np.divide(d-m_hat, np.multiply(p_hat, 1.0-m_hat))
        psi_b_1 = np.multiply(y_resid_d0_weight, y_resid_d0)
        psi_b_2 = np.multiply(1.0-np.divide(d, p_hat), g_hat1 - g_hat0)
        psi_b = psi_b_1 + psi_b_2

    else:
        assert score == 'DR'
        psi_a = -1.0 * np.divide(d, p_hat)
        propensity_weight = np.divide(m_hat, 1.0-m_hat)
        y_resid_d0_weight = np.divide(d, np.mean(d)) \
            - np.divide(np.multiply(1.0-d, propensity_weight), np.mean(np.multiply(1.0-d, propensity_weight)))
        psi_b = np.multiply(y_resid_d0_weight, y_resid_d0)
        
    var = 1/n_obs * np.mean(np.power(np.multiply(psi_a, theta) + psi_b, 2))
    return var


def boot_did(y, d, thetas, ses, all_g_hat0, all_g_hat1, all_m_hat, all_p_hat,
             all_smpls, score, bootstrap, n_rep_boot, dml_procedure,
             n_rep=1, apply_cross_fitting=True):
    all_boot_theta = list()
    all_boot_t_stat = list()
    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]
        if apply_cross_fitting:
            n_obs = len(y)
        else:
            test_index = smpls[0][1]
            n_obs = len(test_index)
        weights = draw_weights(bootstrap, n_rep_boot, n_obs)
        boot_theta, boot_t_stat = boot_did_single_split(
            thetas[i_rep], y, d,
            all_g_hat0[i_rep], all_g_hat1[i_rep], all_m_hat[i_rep], all_p_hat[i_rep], smpls,
            score, ses[i_rep], weights, n_rep_boot, apply_cross_fitting, dml_procedure)
        all_boot_theta.append(boot_theta)
        all_boot_t_stat.append(boot_t_stat)

    boot_theta = np.hstack(all_boot_theta)
    boot_t_stat = np.hstack(all_boot_t_stat)

    return boot_theta, boot_t_stat


def boot_did_single_split(theta, y, d, g_hat0_list, g_hat1_list, m_hat_list, p_hat_list,
                          smpls, score, se, weights, n_rep_boot, apply_cross_fitting, dml_procedure):
    y_resid_d0, g_hat0, g_hat1, m_hat, p_hat = compute_did_residuals(
        y, g_hat0_list, g_hat1_list, m_hat_list, p_hat_list, smpls)

    if apply_cross_fitting:
        if score == 'PA-1':
            J = np.mean(-np.divide(d, p_hat))
        elif score == 'PA-2':
            J = -1.0
        else:
            assert score == 'DR'
            J = np.mean(-np.divide(d, np.mean(d)))
    else:
        test_index = smpls[0][1]
        if score == 'PA-1':
            J = np.mean(-np.divide(d[test_index], p_hat[test_index]))
        elif score == 'PA-2':
            J = -1.0
        else:
            assert score == 'DR'
            J = np.mean(-np.divide(d[test_index], np.mean(d)))

    if score == 'PA-1':
        psi_a = -1.0 * np.divide(d, p_hat)
        y_resid_d0_weight = np.divide(d-m_hat, np.multiply(p_hat, 1.0-m_hat))
        psi_b = np.multiply(y_resid_d0_weight, y_resid_d0)

    elif score == 'PA-2':
        psi_a = -1.0 * np.ones_like(d)
        y_resid_d0_weight = np.divide(d-m_hat, np.multiply(p_hat, 1.0-m_hat))
        psi_b_1 = np.multiply(y_resid_d0_weight, y_resid_d0)
        psi_b_2 = np.multiply(1.0-np.divide(d, p_hat), g_hat1 - g_hat0)
        psi_b = psi_b_1 + psi_b_2

    else:
        assert score == 'DR'
        psi_a = -1.0 * np.divide(d, np.mean(d))
        propensity_weight = np.divide(m_hat, 1.0-m_hat)
        y_resid_d0_weight = np.divide(d, np.mean(d)) \
            - np.divide(np.multiply(1.0-d, propensity_weight), np.mean(np.multiply(1.0-d, propensity_weight)))
        psi_b = np.multiply(y_resid_d0_weight, y_resid_d0)

    psi = np.multiply(psi_a, theta) + psi_b
    boot_theta, boot_t_stat = boot_manual(psi, J, smpls, se, weights, n_rep_boot, apply_cross_fitting)

    return boot_theta, boot_t_stat
