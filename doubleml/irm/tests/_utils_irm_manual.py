import numpy as np
from sklearn.base import clone, is_classifier

from ...tests._utils_boot import boot_manual, draw_weights
from ...tests._utils import fit_predict, fit_predict_proba, tune_grid_search

from ...utils._estimation import _normalize_ipw
from ...utils._checks import _check_is_propensity


def fit_irm(y, x, d,
            learner_g, learner_m, all_smpls, score,
            n_rep=1, g0_params=None, g1_params=None, m_params=None,
            normalize_ipw=True, trimming_threshold=1e-2):
    n_obs = len(y)

    thetas = np.zeros(n_rep)
    ses = np.zeros(n_rep)
    all_g_hat0 = list()
    all_g_hat1 = list()
    all_m_hat = list()
    all_p_hat = list()
    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]

        g_hat0, g_hat1, m_hat, p_hat = fit_nuisance_irm(y, x, d,
                                                        learner_g, learner_m, smpls,
                                                        score,
                                                        g0_params=g0_params, g1_params=g1_params, m_params=m_params,
                                                        trimming_threshold=trimming_threshold)

        all_g_hat0.append(g_hat0)
        all_g_hat1.append(g_hat1)
        all_m_hat.append(m_hat)
        all_p_hat.append(p_hat)

        thetas[i_rep], ses[i_rep] = irm_dml2(y, x, d,
                                             g_hat0, g_hat1, m_hat, p_hat,
                                             smpls, score, normalize_ipw)

    theta = np.median(thetas)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(thetas - theta, 2)) / n_obs)

    res = {'theta': theta, 'se': se,
           'thetas': thetas, 'ses': ses,
           'all_g_hat0': all_g_hat0, 'all_g_hat1': all_g_hat1, 'all_m_hat': all_m_hat, 'all_p_hat': all_p_hat}

    return res


def fit_nuisance_irm(y, x, d, learner_g, learner_m, smpls, score,
                     g0_params=None, g1_params=None, m_params=None,
                     trimming_threshold=1e-12):
    ml_g0 = clone(learner_g)
    ml_g1 = clone(learner_g)
    train_cond0 = np.where(d == 0)[0]
    if is_classifier(learner_g):
        g_hat0_list = fit_predict_proba(y, x, ml_g0, g0_params, smpls,
                                        train_cond=train_cond0)
    else:
        g_hat0_list = fit_predict(y, x, ml_g0, g0_params, smpls,
                                  train_cond=train_cond0)

    train_cond1 = np.where(d == 1)[0]
    if is_classifier(learner_g):
        g_hat1_list = fit_predict_proba(y, x, ml_g1, g1_params, smpls,
                                        train_cond=train_cond1)
    else:
        g_hat1_list = fit_predict(y, x, ml_g1, g1_params, smpls,
                                  train_cond=train_cond1)

    ml_m = clone(learner_m)
    m_hat_list = fit_predict_proba(d, x, ml_m, m_params, smpls,
                                   trimming_threshold=trimming_threshold)

    p_hat_list = []
    for _ in smpls:
        p_hat_list.append(np.mean(d))

    return g_hat0_list, g_hat1_list, m_hat_list, p_hat_list


def tune_nuisance_irm(y, x, d, ml_g, ml_m, smpls, score, n_folds_tune,
                      param_grid_g, param_grid_m):
    train_cond0 = np.where(d == 0)[0]
    g0_tune_res = tune_grid_search(y, x, ml_g, smpls, param_grid_g, n_folds_tune,
                                   train_cond=train_cond0)

    train_cond1 = np.where(d == 1)[0]
    g1_tune_res = tune_grid_search(y, x, ml_g, smpls, param_grid_g, n_folds_tune,
                                   train_cond=train_cond1)

    m_tune_res = tune_grid_search(d, x, ml_m, smpls, param_grid_m, n_folds_tune)

    g0_best_params = [xx.best_params_ for xx in g0_tune_res]
    g1_best_params = [xx.best_params_ for xx in g1_tune_res]
    m_best_params = [xx.best_params_ for xx in m_tune_res]

    return g0_best_params, g1_best_params, m_best_params


def compute_iivm_residuals(y, g_hat0_list, g_hat1_list, m_hat_list, p_hat_list, smpls):
    u_hat0 = np.full_like(y, np.nan, dtype='float64')
    u_hat1 = np.full_like(y, np.nan, dtype='float64')
    g_hat0 = np.full_like(y, np.nan, dtype='float64')
    g_hat1 = np.full_like(y, np.nan, dtype='float64')
    m_hat = np.full_like(y, np.nan, dtype='float64')
    p_hat = np.full_like(y, np.nan, dtype='float64')
    for idx, (_, test_index) in enumerate(smpls):
        u_hat0[test_index] = y[test_index] - g_hat0_list[idx]
        u_hat1[test_index] = y[test_index] - g_hat1_list[idx]
        g_hat0[test_index] = g_hat0_list[idx]
        g_hat1[test_index] = g_hat1_list[idx]
        m_hat[test_index] = m_hat_list[idx]
        p_hat[test_index] = p_hat_list[idx]

    _check_is_propensity(m_hat, 'learner_m', 'ml_m', smpls, eps=1e-12)
    return u_hat0, u_hat1, g_hat0, g_hat1, m_hat, p_hat


def irm_dml2(y, x, d, g_hat0_list, g_hat1_list, m_hat_list, p_hat_list, smpls, score, normalize_ipw):
    n_obs = len(y)
    u_hat0, u_hat1, g_hat0, g_hat1, m_hat, p_hat = compute_iivm_residuals(
        y, g_hat0_list, g_hat1_list, m_hat_list, p_hat_list, smpls)

    if normalize_ipw:
        m_hat_adj = _normalize_ipw(m_hat, d)
    else:
        m_hat_adj = m_hat

    theta_hat = irm_orth(g_hat0, g_hat1, m_hat_adj, p_hat,
                         u_hat0, u_hat1, d, score)
    se = np.sqrt(var_irm(theta_hat, g_hat0, g_hat1,
                         m_hat_adj, p_hat,
                         u_hat0, u_hat1,
                         d, score, n_obs))

    return theta_hat, se


def var_irm(theta, g_hat0, g_hat1, m_hat, p_hat, u_hat0, u_hat1, d, score, n_obs):

    if score == 'ATE':
        var = 1/n_obs * np.mean(np.power(g_hat1 - g_hat0
                                         + np.divide(np.multiply(d, u_hat1), m_hat)
                                         - np.divide(np.multiply(1.-d, u_hat0), 1.-m_hat) - theta, 2))
    else:
        assert score == 'ATTE'
        var = 1/n_obs * np.mean(np.power(np.divide(np.multiply(d, u_hat0), p_hat)
                                         - np.divide(np.multiply(m_hat, np.multiply(1.-d, u_hat0)),
                                                     np.multiply(p_hat, (1.-m_hat)))
                                         - theta * np.divide(d, p_hat), 2)) \
            / np.power(np.mean(np.divide(d, p_hat)), 2)

    return var


def irm_orth(g_hat0, g_hat1, m_hat, p_hat, u_hat0, u_hat1, d, score):

    if score == 'ATE':
        res = np.mean(g_hat1 - g_hat0
                      + np.divide(np.multiply(d, u_hat1), m_hat)
                      - np.divide(np.multiply(1.-d, u_hat0), 1.-m_hat))
    else:
        assert score == 'ATTE'
        res = np.mean(np.divide(np.multiply(d, u_hat0), p_hat)
                      - np.divide(np.multiply(m_hat, np.multiply(1.-d, u_hat0)),
                                  np.multiply(p_hat, (1.-m_hat)))) \
            / np.mean(np.divide(d, p_hat))

    return res


def boot_irm(y, d, thetas, ses, all_g_hat0, all_g_hat1, all_m_hat, all_p_hat,
             all_smpls, score, bootstrap, n_rep_boot,
             n_rep=1, apply_cross_fitting=True, normalize_ipw=True):
    all_boot_t_stat = list()
    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]
        if apply_cross_fitting:
            n_obs = len(y)
        else:
            test_index = smpls[0][1]
            n_obs = len(test_index)
        weights = draw_weights(bootstrap, n_rep_boot, n_obs)
        boot_t_stat = boot_irm_single_split(
            thetas[i_rep], y, d,
            all_g_hat0[i_rep], all_g_hat1[i_rep], all_m_hat[i_rep], all_p_hat[i_rep], smpls,
            score, ses[i_rep], weights, n_rep_boot, apply_cross_fitting, normalize_ipw)
        all_boot_t_stat.append(boot_t_stat)

    boot_t_stat = np.hstack(all_boot_t_stat)

    return boot_t_stat


def boot_irm_single_split(theta, y, d, g_hat0_list, g_hat1_list, m_hat_list, p_hat_list,
                          smpls, score, se, weights, n_rep_boot, apply_cross_fitting, normalize_ipw):
    u_hat0, u_hat1, g_hat0, g_hat1, m_hat, p_hat = compute_iivm_residuals(
        y, g_hat0_list, g_hat1_list, m_hat_list, p_hat_list, smpls)

    m_hat_adj = np.full_like(m_hat, np.nan, dtype='float64')
    if normalize_ipw:
        m_hat_adj = _normalize_ipw(m_hat, d)
    else:
        m_hat_adj = m_hat

    if apply_cross_fitting:
        if score == 'ATE':
            J = -1.0
        else:
            assert score == 'ATTE'
            J = np.mean(-np.divide(d, p_hat))
    else:
        test_index = smpls[0][1]
        if score == 'ATE':
            J = -1.0
        else:
            assert score == 'ATTE'
            J = np.mean(-np.divide(d[test_index], p_hat[test_index]))

    if score == 'ATE':
        psi = g_hat1 - g_hat0 \
                + np.divide(np.multiply(d, u_hat1), m_hat_adj) \
                - np.divide(np.multiply(1.-d, u_hat0), 1.-m_hat_adj) - theta
    else:
        assert score == 'ATTE'
        psi = np.divide(np.multiply(d, u_hat0), p_hat) \
            - np.divide(np.multiply(m_hat_adj, np.multiply(1.-d, u_hat0)),
                        np.multiply(p_hat, (1.-m_hat_adj))) \
            - theta * np.divide(d, p_hat)

    boot_t_stat = boot_manual(psi, J, smpls, se, weights, n_rep_boot, apply_cross_fitting)

    return boot_t_stat


def fit_sensitivity_elements_irm(y, d, all_coef, predictions, score, n_rep):
    n_treat = 1
    n_obs = len(y)

    sigma2 = np.full(shape=(1, n_rep, n_treat), fill_value=np.nan)
    nu2 = np.full(shape=(1, n_rep, n_treat), fill_value=np.nan)
    psi_sigma2 = np.full(shape=(n_obs, n_rep, n_treat), fill_value=np.nan)
    psi_nu2 = np.full(shape=(n_obs, n_rep, n_treat), fill_value=np.nan)

    for i_rep in range(n_rep):

        m_hat = predictions['ml_m'][:, i_rep, 0]
        g_hat0 = predictions['ml_g0'][:, i_rep, 0]
        if score == 'ATE':
            g_hat1 = predictions['ml_g1'][:, i_rep, 0]
        else:
            assert score == 'ATTE'
            g_hat1 = y

        if score == 'ATE':
            weights = np.ones_like(d)
            weights_bar = np.ones_like(d)
        else:
            assert score == 'ATTE'
            weights = np.divide(d, np.mean(d))
            weights_bar = np.divide(m_hat, np.mean(d))

        sigma2_score_element = np.square(y - np.multiply(d, g_hat1) - np.multiply(1.0-d, g_hat0))
        sigma2[0, i_rep, 0] = np.mean(sigma2_score_element)
        psi_sigma2[:, i_rep, 0] = sigma2_score_element - sigma2[0, i_rep, 0]

        # calc m(W,alpha) and Riesz representer
        m_alpha = np.multiply(weights, np.multiply(weights_bar, (np.divide(1.0, m_hat) + np.divide(1.0, 1.0-m_hat))))
        rr = np.multiply(weights_bar, (np.divide(d, m_hat) - np.divide(1.0-d, 1.0-m_hat)))

        nu2_score_element = np.multiply(2.0, m_alpha) - np.square(rr)
        nu2[0, i_rep, 0] = np.mean(nu2_score_element)
        psi_nu2[:, i_rep, 0] = nu2_score_element - nu2[0, i_rep, 0]

    element_dict = {'sigma2': sigma2,
                    'nu2': nu2,
                    'psi_sigma2': psi_sigma2,
                    'psi_nu2': psi_nu2}
    return element_dict
