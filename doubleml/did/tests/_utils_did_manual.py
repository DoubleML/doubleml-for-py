import numpy as np
from sklearn.base import clone

from ...tests._utils_boot import boot_manual, draw_weights
from ...tests._utils import fit_predict, fit_predict_proba, tune_grid_search


def fit_did(y, x, d,
            learner_g, learner_m, all_smpls, score, in_sample_normalization,
            n_rep=1, g0_params=None, g1_params=None, m_params=None,
            trimming_threshold=1e-2):
    n_obs = len(y)

    thetas = np.zeros(n_rep)
    ses = np.zeros(n_rep)
    all_g_hat0 = list()
    all_g_hat1 = list()
    all_m_hat = list()
    all_p_hat = list()
    all_psi_a = list()
    all_psi_b = list()
    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]

        g_hat0_list, g_hat1_list, m_hat_list, \
            p_hat_list = fit_nuisance_did(y, x, d,
                                          learner_g, learner_m, smpls,
                                          score,
                                          g0_params=g0_params, g1_params=g1_params, m_params=m_params,
                                          trimming_threshold=trimming_threshold)

        all_g_hat0.append(g_hat0_list)
        all_g_hat1.append(g_hat1_list)
        all_m_hat.append(m_hat_list)
        all_p_hat.append(p_hat_list)

        resid_d0, g_hat0, g_hat1, m_hat, p_hat = compute_did_residuals(
            y, g_hat0_list, g_hat1_list, m_hat_list, p_hat_list, smpls)

        psi_a, psi_b = did_score_elements(g_hat0, g_hat1, m_hat, p_hat,
                                          resid_d0, d, score, in_sample_normalization)

        all_psi_a.append(psi_a)
        all_psi_b.append(psi_b)

        thetas[i_rep], ses[i_rep] = did_dml2(psi_a, psi_b)

    theta = np.median(thetas)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(thetas - theta, 2)) / n_obs)

    res = {'theta': theta, 'se': se,
           'thetas': thetas, 'ses': ses,
           'all_g_hat0': all_g_hat0, 'all_g_hat1': all_g_hat1, 'all_m_hat': all_m_hat, 'all_p_hat': all_p_hat,
           'all_psi_a': all_psi_a, 'all_psi_b': all_psi_b}

    return res


def fit_nuisance_did(y, x, d, learner_g, learner_m, smpls, score,
                     g0_params=None, g1_params=None, m_params=None,
                     trimming_threshold=1e-12):
    ml_g0 = clone(learner_g)
    ml_g1 = clone(learner_g)
    train_cond0 = np.where(d == 0)[0]
    g_hat0_list = fit_predict(y, x, ml_g0, g0_params, smpls,
                              train_cond=train_cond0)

    train_cond1 = np.where(d == 1)[0]
    g_hat1_list = fit_predict(y, x, ml_g1, g1_params, smpls,
                              train_cond=train_cond1)
    if score == 'experimental':
        m_hat_list = list()
        for idx, _ in enumerate(smpls):
            # fill it up, but its not further used
            m_hat_list.append(np.zeros_like(g_hat0_list[idx], dtype='float64'))

    else:
        assert score == 'observational'
        ml_m = clone(learner_m)
        m_hat_list = fit_predict_proba(d, x, ml_m, m_params, smpls,
                                       trimming_threshold=trimming_threshold)

    p_hat_list = []
    for (train_index, _) in smpls:
        p_hat_list.append(np.mean(d[train_index]))

    return g_hat0_list, g_hat1_list, m_hat_list, p_hat_list


def compute_did_residuals(y, g_hat0_list, g_hat1_list, m_hat_list, p_hat_list, smpls):
    resid_d0 = np.full_like(y, np.nan, dtype='float64')
    g_hat0 = np.full_like(y, np.nan, dtype='float64')
    g_hat1 = np.full_like(y, np.nan, dtype='float64')
    m_hat = np.full_like(y, np.nan, dtype='float64')
    p_hat = np.full_like(y, np.nan, dtype='float64')
    for idx, (_, test_index) in enumerate(smpls):
        resid_d0[test_index] = y[test_index] - g_hat0_list[idx]
        g_hat0[test_index] = g_hat0_list[idx]
        g_hat1[test_index] = g_hat1_list[idx]
        m_hat[test_index] = m_hat_list[idx]
        p_hat[test_index] = p_hat_list[idx]

    return resid_d0, g_hat0, g_hat1, m_hat, p_hat


def did_dml2(psi_a, psi_b):
    n_obs = len(psi_a)
    theta_hat = - np.mean(psi_b) / np.mean(psi_a)
    se = np.sqrt(var_did(theta_hat, psi_a, psi_b, n_obs))

    return theta_hat, se


def did_score_elements(g_hat0, g_hat1, m_hat, p_hat, resid_d0, d, score, in_sample_normalization):

    if score == 'observational':
        if in_sample_normalization:
            weight_psi_a = np.divide(d, np.mean(d))
            propensity_weight = np.multiply(1.0-d, np.divide(m_hat, 1.0-m_hat))
            weight_resid_d0 = np.divide(d, np.mean(d)) - np.divide(propensity_weight, np.mean(propensity_weight))
        else:
            weight_psi_a = np.divide(d, p_hat)
            weight_resid_d0 = np.divide(d-m_hat, np.multiply(p_hat, 1.0-m_hat))

        psi_b_1 = np.zeros_like(d)

    else:
        assert score == 'experimental'
        if in_sample_normalization:
            weight_psi_a = np.ones_like(d)
            weight_g0 = np.divide(d, np.mean(d)) - 1.0
            weight_g1 = 1.0 - np.divide(d, np.mean(d))
            weight_resid_d0 = np.divide(d, np.mean(d)) - np.divide(1.0-d, np.mean(1.0-d))
        else:
            weight_psi_a = np.ones_like(d)
            weight_g0 = np.divide(d, p_hat) - 1.0
            weight_g1 = 1.0 - np.divide(d, p_hat)
            weight_resid_d0 = np.divide(d-p_hat, np.multiply(p_hat, 1.0-p_hat))

        psi_b_1 = np.multiply(weight_g0,  g_hat0) + np.multiply(weight_g1,  g_hat1)

    psi_a = -1.0 * weight_psi_a
    psi_b = psi_b_1 + np.multiply(weight_resid_d0,  resid_d0)

    return psi_a, psi_b


def var_did(theta, psi_a, psi_b, n_obs):
    J = np.mean(psi_a)
    var = 1/n_obs * np.mean(np.power(np.multiply(psi_a, theta) + psi_b, 2)) / np.power(J, 2)
    return var


def boot_did(y, thetas, ses, all_psi_a, all_psi_b,
             all_smpls, bootstrap, n_rep_boot, n_rep=1, apply_cross_fitting=True):
    all_boot_t_stat = list()
    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]
        if apply_cross_fitting:
            n_obs = len(y)
        else:
            test_index = smpls[0][1]
            n_obs = len(test_index)
        weights = draw_weights(bootstrap, n_rep_boot, n_obs)
        boot_t_stat = boot_did_single_split(
            thetas[i_rep], all_psi_a[i_rep], all_psi_b[i_rep], smpls,
            ses[i_rep], weights, n_rep_boot, apply_cross_fitting)
        all_boot_t_stat.append(boot_t_stat)

    boot_t_stat = np.hstack(all_boot_t_stat)

    return boot_t_stat


def boot_did_single_split(theta, psi_a, psi_b,
                          smpls, se, weights, n_rep_boot, apply_cross_fitting):

    if apply_cross_fitting:
        J = np.mean(psi_a)
    else:
        test_index = smpls[0][1]
        J = np.mean(psi_a[test_index])

    psi = np.multiply(psi_a, theta) + psi_b
    boot_t_stat = boot_manual(psi, J, smpls, se, weights, n_rep_boot, apply_cross_fitting)

    return boot_t_stat


def tune_nuisance_did(y, x, d, ml_g, ml_m, smpls, score, n_folds_tune,
                      param_grid_g, param_grid_m):
    train_cond0 = np.where(d == 0)[0]
    g0_tune_res = tune_grid_search(y, x, ml_g, smpls, param_grid_g, n_folds_tune,
                                   train_cond=train_cond0)
    g0_best_params = [xx.best_params_ for xx in g0_tune_res]

    train_cond1 = np.where(d == 1)[0]
    g1_tune_res = tune_grid_search(y, x, ml_g, smpls, param_grid_g, n_folds_tune,
                                   train_cond=train_cond1)
    g1_best_params = [xx.best_params_ for xx in g1_tune_res]

    if score == 'experimental':
        m_best_params = None
    else:
        assert score == 'observational'
        m_tune_res = tune_grid_search(d, x, ml_m, smpls, param_grid_m, n_folds_tune)
        m_best_params = [xx.best_params_ for xx in m_tune_res]

    return g0_best_params, g1_best_params, m_best_params


def fit_sensitivity_elements_did(y, d, all_coef, predictions, score, in_sample_normalization, n_rep):
    n_treat = 1
    n_obs = len(y)

    sigma2 = np.full(shape=(1, n_rep, n_treat), fill_value=np.nan)
    nu2 = np.full(shape=(1, n_rep, n_treat), fill_value=np.nan)
    psi_sigma2 = np.full(shape=(n_obs, n_rep, n_treat), fill_value=np.nan)
    psi_nu2 = np.full(shape=(n_obs, n_rep, n_treat), fill_value=np.nan)

    for i_rep in range(n_rep):

        m_hat = predictions['ml_m'][:, i_rep, 0]
        g_hat0 = predictions['ml_g0'][:, i_rep, 0]
        g_hat1 = predictions['ml_g1'][:, i_rep, 0]

        sigma2_score_element = np.square(y - np.multiply(d, g_hat1) - np.multiply(1.0-d, g_hat0))
        sigma2[0, i_rep, 0] = np.mean(sigma2_score_element)
        psi_sigma2[:, i_rep, 0] = sigma2_score_element - sigma2[0, i_rep, 0]

        if score == 'observational':
            propensity_weight_d0 = np.divide(m_hat, 1.0-m_hat)
            if in_sample_normalization:
                m_alpha_1 = np.divide(d, np.mean(d))
                m_alpha_2 = np.divide(1, np.mean(d)) + \
                    np.divide(propensity_weight_d0, np.mean(np.multiply(1.0-d, propensity_weight_d0)))
                m_alpha = np.multiply(m_alpha_1, m_alpha_2)
                rr_1 = np.multiply(1.0-d, propensity_weight_d0)
                rr = np.divide(d, np.mean(d)) - np.divide(rr_1, np.mean(rr_1))
            else:
                m_alpha_1 = np.divide(d, np.square(np.mean(d)))
                m_alpha = np.multiply(m_alpha_1, 1.0+propensity_weight_d0)
                rr = np.divide(d, np.mean(d)) - np.multiply(np.divide(1.0-d, np.mean(d)), propensity_weight_d0)
        else:
            assert score == 'experimental'
            m_alpha = np.divide(1.0, np.mean(d)) + np.divide(1.0, 1.0-np.mean(d))
            rr = np.divide(d, np.mean(d)) - np.divide(1.0-d, 1.0-np.mean(d))

        nu2_score_element = np.multiply(2.0, m_alpha) - np.square(rr)
        nu2[0, i_rep, 0] = np.mean(nu2_score_element)
        psi_nu2[:, i_rep, 0] = nu2_score_element - nu2[0, i_rep, 0]

    element_dict = {'sigma2': sigma2,
                    'nu2': nu2,
                    'psi_sigma2': psi_sigma2,
                    'psi_nu2': psi_nu2}
    return element_dict
