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
    var = 1/n_obs * np.mean(np.power(g_hat1 + np.divide(np.multiply(treated, u_hat1), m_hat) - theta, 2))
    return var


def boot_apo(y, d, treatment_level, thetas, ses, all_g_hat0, all_g_hat1, all_m_hat,
             all_smpls, score, bootstrap, n_rep_boot,
             n_rep=1, normalize_ipw=True):
    treated = (d == treatment_level)
    all_boot_t_stat = list()
    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]
        n_obs = len(y)

        weights = draw_weights(bootstrap, n_rep_boot, n_obs)
        boot_t_stat = boot_apo_single_split(
            thetas[i_rep], y, d, treated,
            all_g_hat0[i_rep], all_g_hat1[i_rep], all_m_hat[i_rep], smpls,
            score, ses[i_rep], weights, n_rep_boot, normalize_ipw)
        all_boot_t_stat.append(boot_t_stat)

    boot_t_stat = np.hstack(all_boot_t_stat)

    return boot_t_stat


def boot_apo_single_split(theta, y, d, treated, g_hat0_list, g_hat1_list, m_hat_list,
                          smpls, score, se, weights, n_rep_boot, normalize_ipw):
    _, u_hat1, _, g_hat1, m_hat = compute_residuals(
        y, g_hat0_list, g_hat1_list, m_hat_list, smpls)

    if normalize_ipw:
        m_hat_adj = _normalize_ipw(m_hat, treated)
    else:
        m_hat_adj = m_hat

    J = -1.0
    psi = g_hat1 + np.divide(np.multiply(treated, u_hat1), m_hat_adj) - theta
    boot_t_stat = boot_manual(psi, J, smpls, se, weights, n_rep_boot)

    return boot_t_stat


def fit_sensitivity_elements_apo(y, d, treatment_level, all_coef, predictions, score, n_rep):
    n_treat = 1
    n_obs = len(y)
    treated = (d == treatment_level)

    sigma2 = np.full(shape=(1, n_rep, n_treat), fill_value=np.nan)
    nu2 = np.full(shape=(1, n_rep, n_treat), fill_value=np.nan)
    psi_sigma2 = np.full(shape=(n_obs, n_rep, n_treat), fill_value=np.nan)
    psi_nu2 = np.full(shape=(n_obs, n_rep, n_treat), fill_value=np.nan)

    for i_rep in range(n_rep):

        m_hat = predictions['ml_m'][:, i_rep, 0]
        g_hat0 = predictions['ml_g0'][:, i_rep, 0]
        g_hat1 = predictions['ml_g1'][:, i_rep, 0]

        weights = np.ones_like(d)
        weights_bar = np.ones_like(d)

        sigma2_score_element = np.square(y - np.multiply(treated, g_hat1) - np.multiply(1.0-treated, g_hat0))
        sigma2[0, i_rep, 0] = np.mean(sigma2_score_element)
        psi_sigma2[:, i_rep, 0] = sigma2_score_element - sigma2[0, i_rep, 0]

        # calc m(W,alpha) and Riesz representer
        m_alpha = np.multiply(weights, np.multiply(weights_bar, np.divide(1.0, m_hat)))
        rr = np.multiply(weights_bar, np.divide(treated, m_hat))

        nu2_score_element = np.multiply(2.0, m_alpha) - np.square(rr)
        nu2[0, i_rep, 0] = np.mean(nu2_score_element)
        psi_nu2[:, i_rep, 0] = nu2_score_element - nu2[0, i_rep, 0]

    element_dict = {'sigma2': sigma2,
                    'nu2': nu2,
                    'psi_sigma2': psi_sigma2,
                    'psi_nu2': psi_nu2}
    return element_dict


def tune_nuisance_apo(y, x, d, treatment_level, ml_g, ml_m, smpls, score, n_folds_tune,
                      param_grid_g, param_grid_m):
    train_cond0 = np.where(d != treatment_level)[0]
    g0_tune_res = tune_grid_search(y, x, ml_g, smpls, param_grid_g, n_folds_tune,
                                   train_cond=train_cond0)

    train_cond1 = np.where(d == treatment_level)[0]
    g1_tune_res = tune_grid_search(y, x, ml_g, smpls, param_grid_g, n_folds_tune,
                                   train_cond=train_cond1)

    treated = (d == treatment_level)
    m_tune_res = tune_grid_search(treated, x, ml_m, smpls, param_grid_m, n_folds_tune)

    g0_best_params = [xx.best_params_ for xx in g0_tune_res]
    g1_best_params = [xx.best_params_ for xx in g1_tune_res]
    m_best_params = [xx.best_params_ for xx in m_tune_res]

    return g0_best_params, g1_best_params, m_best_params
