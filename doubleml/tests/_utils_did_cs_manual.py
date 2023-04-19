import numpy as np
from sklearn.base import clone

from ._utils import fit_predict, fit_predict_proba, tune_grid_search
from ._utils_did_manual import did_dml1, did_dml2

from .._utils import _check_is_propensity


def fit_did_cs(y, x, d, t,
               learner_g, learner_m, all_smpls, dml_procedure, score, in_sample_normalization,
               n_rep=1, g_d0_t0_params=None, g_d0_t1_params=None,
               g_d1_t0_params=None, g_d1_t1_params=None, m_params=None,
               trimming_threshold=1e-2):
    n_obs = len(y)

    thetas = np.zeros(n_rep)
    ses = np.zeros(n_rep)
    all_g_hat_d0_t0 = list()
    all_g_hat_d0_t1 = list()
    all_g_hat_d1_t0 = list()
    all_g_hat_d1_t1 = list()
    all_m_hat = list()
    all_p_hat = list()
    all_lambda_hat = list()
    all_psi_a = list()
    all_psi_b = list()
    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]

        g_hat_d0_t0_list, g_hat_d0_t1_list, g_hat_d1_t0_list, g_hat_d1_t1_list, m_hat_list, \
            p_hat_list,  lambda_hat_list = fit_nuisance_did_cs(y, x, d, t,
                                                               learner_g, learner_m,
                                                               smpls, score,
                                                               g_d0_t0_params=g_d0_t0_params, g_d0_t1_params=g_d0_t1_params,
                                                               g_d1_t0_params=g_d1_t0_params, g_d1_t1_params=g_d1_t1_params,
                                                               m_params=m_params,
                                                               trimming_threshold=trimming_threshold)

        all_g_hat_d0_t0.append(g_hat_d0_t0_list)
        all_g_hat_d0_t1.append(g_hat_d0_t1_list)
        all_g_hat_d1_t0.append(g_hat_d1_t0_list)
        all_g_hat_d1_t1.append(g_hat_d1_t1_list)
        all_m_hat.append(m_hat_list)
        all_p_hat.append(p_hat_list)
        all_lambda_hat.append(lambda_hat_list)

        resid_d0_t0, resid_d0_t1, resid_d1_t0, resid_d1_t1, \
            g_hat_d0_t0, g_hat_d0_t1, g_hat_d1_t0, g_hat_d1_t1, \
            m_hat, p_hat, lambda_hat = compute_did_cs_residuals(y, g_hat_d0_t0_list, g_hat_d0_t1_list,
                                                                g_hat_d1_t0_list, g_hat_d1_t1_list,
                                                                m_hat_list, p_hat_list,
                                                                lambda_hat_list, smpls)

        psi_a, psi_b = did_cs_score_elements(resid_d0_t0, resid_d0_t1, resid_d1_t0, resid_d1_t1,
                                             g_hat_d0_t0, g_hat_d0_t1, g_hat_d1_t0, g_hat_d1_t1,
                                             m_hat, p_hat, lambda_hat, d, t, score, in_sample_normalization)

        all_psi_a.append(psi_a)
        all_psi_b.append(psi_b)

        if dml_procedure == 'dml1':
            thetas[i_rep], ses[i_rep] = did_dml1(psi_a, psi_b, smpls)
        else:
            assert dml_procedure == 'dml2'
            thetas[i_rep], ses[i_rep] = did_dml2(psi_a, psi_b)

    theta = np.median(thetas)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(thetas - theta, 2)) / n_obs)

    res = {'theta': theta, 'se': se,
           'thetas': thetas, 'ses': ses,
           'all_g_hat_d0_t0': all_g_hat_d0_t0, 'all_g_hat_d0_t1': all_g_hat_d0_t1,
           'all_g_hat_d1_t0': all_g_hat_d1_t0, 'all_g_hat_d1_t1': all_g_hat_d1_t1,
           'all_m_hat': all_m_hat,
           'all_p_hat': all_p_hat, 'all_lambda_hat': all_lambda_hat,
           'all_psi_a': all_psi_a, 'all_psi_b': all_psi_b}

    return res


def fit_nuisance_did_cs(y, x, d, t,
                        learner_g, learner_m, smpls, score,
                        g_d0_t0_params=None, g_d0_t1_params=None,
                        g_d1_t0_params=None, g_d1_t1_params=None,
                        m_params=None,
                        trimming_threshold=1e-12):
    ml_g_d0_t0 = clone(learner_g)
    ml_g_d0_t1 = clone(learner_g)
    ml_g_d1_t0 = clone(learner_g)
    ml_g_d1_t1 = clone(learner_g)

    train_cond_d0_t0 = np.intersect1d(np.where(d == 0)[0], np.where(t == 0)[0])
    g_hat_d0_t0_list = fit_predict(y, x, ml_g_d0_t0, g_d0_t0_params, smpls,
                                   train_cond=train_cond_d0_t0)

    train_cond_d0_t1 = np.intersect1d(np.where(d == 0)[0], np.where(t == 1)[0])
    g_hat_d0_t1_list = fit_predict(y, x, ml_g_d0_t1, g_d0_t1_params, smpls,
                                   train_cond=train_cond_d0_t1)

    train_cond_d1_t0 = np.intersect1d(np.where(d == 1)[0], np.where(t == 0)[0])
    g_hat_d1_t0_list = fit_predict(y, x, ml_g_d1_t0, g_d1_t0_params, smpls,
                                   train_cond=train_cond_d1_t0)

    train_cond_d1_t1 = np.intersect1d(np.where(d == 1)[0], np.where(t == 1)[0])
    g_hat_d1_t1_list = fit_predict(y, x, ml_g_d1_t1, g_d1_t1_params, smpls,
                                   train_cond=train_cond_d1_t1)

    ml_m = clone(learner_m)
    m_hat_list = fit_predict_proba(d, x, ml_m, m_params, smpls,
                                   trimming_threshold=trimming_threshold)

    p_hat_list = []
    for (train_index, _) in smpls:
        p_hat_list.append(np.mean(d[train_index]))

    lambda_hat_list = []
    for (train_index, _) in smpls:
        lambda_hat_list.append(np.mean(t[train_index]))

    return g_hat_d0_t0_list, g_hat_d0_t1_list, g_hat_d1_t0_list, g_hat_d1_t1_list, \
        m_hat_list, p_hat_list,  lambda_hat_list


def compute_did_cs_residuals(y, g_hat_d0_t0_list, g_hat_d0_t1_list,
                             g_hat_d1_t0_list, g_hat_d1_t1_list,
                             m_hat_list, p_hat_list, lambda_hat_list, smpls):
    g_hat_d0_t0 = np.full_like(y, np.nan, dtype='float64')
    g_hat_d0_t1 = np.full_like(y, np.nan, dtype='float64')
    g_hat_d1_t0 = np.full_like(y, np.nan, dtype='float64')
    g_hat_d1_t1 = np.full_like(y, np.nan, dtype='float64')
    m_hat = np.full_like(y, np.nan, dtype='float64')
    p_hat = np.full_like(y, np.nan, dtype='float64')
    lambda_hat = np.full_like(y, np.nan, dtype='float64')
    for idx, (_, test_index) in enumerate(smpls):
        g_hat_d0_t0[test_index] = g_hat_d0_t0_list[idx]
        g_hat_d0_t1[test_index] = g_hat_d0_t1_list[idx]
        g_hat_d1_t0[test_index] = g_hat_d1_t0_list[idx]
        g_hat_d1_t1[test_index] = g_hat_d1_t1_list[idx]
        m_hat[test_index] = m_hat_list[idx]
        p_hat[test_index] = p_hat_list[idx]
        lambda_hat[test_index] = lambda_hat_list[idx]

    resid_d0_t0 = y - g_hat_d0_t0
    resid_d0_t1 = y - g_hat_d0_t1
    resid_d1_t0 = y - g_hat_d1_t0
    resid_d1_t1 = y - g_hat_d1_t1
    _check_is_propensity(m_hat, 'learner_m', 'ml_m', smpls, eps=1e-12)
    return resid_d0_t0, resid_d0_t1, resid_d1_t0, resid_d1_t1, \
        g_hat_d0_t0, g_hat_d0_t1, g_hat_d1_t0, g_hat_d1_t1, \
        m_hat, p_hat, lambda_hat


def did_cs_score_elements(resid_d0_t0, resid_d0_t1, resid_d1_t0, resid_d1_t1,
                          g_hat_d0_t0, g_hat_d0_t1, g_hat_d1_t0, g_hat_d1_t1,
                          m_hat, p_hat, lambda_hat, d, t, score, in_sample_normalization):

    if score == 'observational':
        if in_sample_normalization:
            weight_psi_a = np.ones_like(d)
            weight_g_d1_t1 = np.divide(d, np.mean(d))
            weight_g_d1_t0 = -1.0 * weight_g_d1_t1
            weight_g_d0_t1 = -1.0 * weight_g_d1_t1
            weight_g_d0_t0 = weight_g_d1_t1

            weight_resid_d1_t1 = np.divide(np.multiply(d, t),
                                           np.mean(np.multiply(d, t)))
            weight_resid_d1_t0 = -1.0 * np.divide(np.multiply(d, 1.0-t),
                                                  np.mean(np.multiply(d, 1.0-t)))

            prop_weighting = np.divide(m_hat, 1.0-m_hat)
            unscaled_d0_t1 = np.multiply(np.multiply(1.0-d, t), prop_weighting)
            weight_resid_d0_t1 = -1.0 * np.divide(unscaled_d0_t1, np.mean(unscaled_d0_t1))

            unscaled_d0_t0 = np.multiply(np.multiply(1.0-d, 1.0-t), prop_weighting)
            weight_resid_d0_t0 = np.divide(unscaled_d0_t0, np.mean(unscaled_d0_t0))
        else:
            weight_psi_a = np.divide(d, p_hat)
            weight_g_d1_t1 = weight_psi_a
            weight_g_d1_t0 = -1.0 * weight_psi_a
            weight_g_d0_t1 = -1.0 * weight_psi_a
            weight_g_d0_t0 = weight_psi_a

            weight_resid_d1_t1 = np.divide(np.multiply(d, t),
                                           np.multiply(p_hat, lambda_hat))
            weight_resid_d1_t0 = -1.0 * np.divide(np.multiply(d, 1.0-t),
                                                  np.multiply(p_hat, 1.0-lambda_hat))

            prop_weighting = np.divide(m_hat, 1.0-m_hat)
            weight_resid_d0_t1 = -1.0 * np.multiply(np.divide(np.multiply(1.0-d, t),
                                                              np.multiply(p_hat, lambda_hat)),
                                                    prop_weighting)
            weight_resid_d0_t0 = np.multiply(np.divide(np.multiply(1.0-d, 1.0-t),
                                                       np.multiply(p_hat, 1.0-lambda_hat)),
                                             prop_weighting)

    else:
        assert score == 'experimental'
        if in_sample_normalization:
            weight_psi_a = np.ones_like(d)
            weight_g_d1_t1 = weight_psi_a
            weight_g_d1_t0 = -1.0 * weight_psi_a
            weight_g_d0_t1 = -1.0 * weight_psi_a
            weight_g_d0_t0 = weight_psi_a

            weight_resid_d1_t1 = np.divide(np.multiply(d, t),
                                           np.mean(np.multiply(d, t)))
            weight_resid_d1_t0 = -1.0 * np.divide(np.multiply(d, 1.0-t),
                                                  np.mean(np.multiply(d, 1.0-t)))
            weight_resid_d0_t1 = -1.0 * np.divide(np.multiply(1.0-d, t),
                                                  np.mean(np.multiply(1.0-d, t)))
            weight_resid_d0_t0 = np.divide(np.multiply(1.0-d, 1.0-t),
                                           np.mean(np.multiply(1.0-d, 1.0-t)))
        else:
            weight_psi_a = np.ones_like(d)
            weight_g_d1_t1 = weight_psi_a
            weight_g_d1_t0 = -1.0 * weight_psi_a
            weight_g_d0_t1 = -1.0 * weight_psi_a
            weight_g_d0_t0 = weight_psi_a

            weight_resid_d1_t1 = np.divide(np.multiply(d, t),
                                           np.multiply(p_hat, lambda_hat))
            weight_resid_d1_t0 = -1.0 * np.divide(np.multiply(d, 1.0-t),
                                                  np.multiply(p_hat, 1.0-lambda_hat))
            weight_resid_d0_t1 = -1.0 * np.divide(np.multiply(1.0-d, t),
                                                  np.multiply(1.0-p_hat, lambda_hat))
            weight_resid_d0_t0 = np.divide(np.multiply(1.0-d, 1.0-t),
                                           np.multiply(1.0-p_hat, 1.0-lambda_hat))

    psi_b_1 = np.multiply(weight_g_d1_t1,  g_hat_d1_t1) + \
        np.multiply(weight_g_d1_t0,  g_hat_d1_t0) + \
        np.multiply(weight_g_d0_t0,  g_hat_d0_t0) + \
        np.multiply(weight_g_d0_t1,  g_hat_d0_t1)
    psi_b_2 = np.multiply(weight_resid_d1_t1,  resid_d1_t1) + \
        np.multiply(weight_resid_d1_t0,  resid_d1_t0) + \
        np.multiply(weight_resid_d0_t0,  resid_d0_t0) + \
        np.multiply(weight_resid_d0_t1,  resid_d0_t1)

    psi_a = -1.0 * weight_psi_a
    psi_b = psi_b_1 + psi_b_2

    return psi_a, psi_b


def tune_nuisance_did_cs(y, x, d, t, ml_g, ml_m, smpls, score, n_folds_tune,
                         param_grid_g, param_grid_m):

    smpls_d0_t0 = np.intersect1d(np.where(d == 0)[0], np.where(t == 0)[0])
    smpls_d0_t1 = np.intersect1d(np.where(d == 0)[0], np.where(t == 1)[0])
    smpls_d1_t0 = np.intersect1d(np.where(d == 1)[0], np.where(t == 0)[0])
    smpls_d1_t1 = np.intersect1d(np.where(d == 1)[0], np.where(t == 1)[0])

    g_d0_t0_tune_res = tune_grid_search(y, x, ml_g, smpls, param_grid_g, n_folds_tune,
                                        train_cond=smpls_d0_t0)
    g_d0_t1_tune_res = tune_grid_search(y, x, ml_g, smpls, param_grid_g, n_folds_tune,
                                        train_cond=smpls_d0_t1)
    g_d1_t0_tune_res = tune_grid_search(y, x, ml_g, smpls, param_grid_g, n_folds_tune,
                                        train_cond=smpls_d1_t0)
    g_d1_t1_tune_res = tune_grid_search(y, x, ml_g, smpls, param_grid_g, n_folds_tune,
                                        train_cond=smpls_d1_t1)

    m_tune_res = tune_grid_search(d, x, ml_m, smpls, param_grid_m, n_folds_tune)

    g_d0_t0_best_params = [xx.best_params_ for xx in g_d0_t0_tune_res]
    g_d0_t1_best_params = [xx.best_params_ for xx in g_d0_t1_tune_res]
    g_d1_t0_best_params = [xx.best_params_ for xx in g_d1_t0_tune_res]
    g_d1_t1_best_params = [xx.best_params_ for xx in g_d1_t1_tune_res]
    m_best_params = [xx.best_params_ for xx in m_tune_res]

    return g_d0_t0_best_params, g_d0_t1_best_params, \
        g_d1_t0_best_params, g_d1_t1_best_params, m_best_params