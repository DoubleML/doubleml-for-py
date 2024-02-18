import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from ._utils_boot import boot_manual, draw_weights
from ._utils import fit_predict, tune_grid_search, fit_predict_proba
from .._utils import _trimm


def fit_selection(y, x, d, z, s,
               learner_mu, learner_pi, learner_p, 
               all_smpls, dml_procedure, score,
               trimming_rule='truncate',
               trimming_threshold=1e-2,
               normalize_ipw=True,
               n_rep=1, 
               mu_d0_params=None, mu_d1_params=None,
               pi_d0_params=None, pi_d1_params=None, 
               p_d0_params=None, p_d1_params=None):
    n_obs = len(y)

    thetas = np.zeros(n_rep)
    ses = np.zeros(n_rep)

    all_mu_d1_hat = list()
    all_mu_d0_hat = list()
    all_pi_d1_hat = list()
    all_pi_d0_hat = list()
    all_p_d1_hat = list()
    all_p_d0_hat = list()

    all_psi_a = list()
    all_psi_b = list()

    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]

        mu_hat_d1_list, mu_hat_d0_list, pi_hat_d1_list, pi_hat_d0_list, \
            p_hat_d1_list, p_hat_d0_list = fit_nuisance_selection(y, x, d, z, s,
                                                        learner_mu, learner_pi, learner_p,
                                                        smpls, score,
                                                        trimming_rule=trimming_rule,
                                                        trimming_threshold=trimming_threshold,
                                                        mu_d0_params=mu_d0_params, mu_d1_params=mu_d1_params,
                                                        pi_d0_params=pi_d0_params, pi_d1_params=pi_d1_params, 
                                                        p_d0_params=p_d0_params, p_d1_params=p_d1_params)
        all_mu_d1_hat.append(mu_hat_d1_list)
        all_mu_d0_hat.append(mu_hat_d0_list)
        all_pi_d1_hat.append(pi_hat_d1_list)
        all_pi_d0_hat.append(pi_hat_d0_list)
        all_p_d1_hat.append(p_hat_d1_list)
        all_p_d0_hat.append(p_hat_d0_list)

        dtreat = (d == 1)
        dcontrol = (d == 0)

        mu_hat_d1, mu_hat_d0, pi_hat_d1, pi_hat_d0, \
            p_hat_d1, p_hat_d0 = compute_selection(y, mu_hat_d1_list, mu_hat_d0_list, pi_hat_d1_list, pi_hat_d0_list,\
                                                    p_hat_d1_list, p_hat_d0_list, smpls)
        
        psi_a, psi_b = selection_score_elements(dtreat, dcontrol, mu_hat_d1, mu_hat_d0, pi_hat_d1, pi_hat_d0,
                                                p_hat_d1, p_hat_d0, s, y, normalize_ipw, smpls)
        
        all_psi_a.append(psi_a)
        all_psi_b.append(psi_b)

        if dml_procedure == 'dml1':
            thetas[i_rep], ses[i_rep] = selection_dml1(psi_a, psi_b, smpls)
        else:
            assert dml_procedure == 'dml2'
            thetas[i_rep], ses[i_rep] = selection_dml2(psi_a, psi_b)
        
    theta = np.median(thetas)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(thetas - theta, 2)) / n_obs)

    res = {'theta': theta, 'se': se,
           'thetas': thetas, 'ses': ses,
           'all_mu_d1_hat': all_mu_d1_hat, 'all_mu_d0_hat': all_mu_d0_hat,
           'all_pi_d1_hat': all_pi_d1_hat, 'all_pi_d0_hat': all_pi_d0_hat,
           'all_p_d1_hat': all_p_d1_hat, 'all_p_d0_hat': all_p_d0_hat,
           'all_psi_a': all_psi_a, 'all_psi_b': all_psi_b}

    return res


def fit_nuisance_selection(y, x, d, z, s,
               learner_mu, learner_pi, learner_p, 
               smpls, score,
               trimming_rule='truncate',
               trimming_threshold=1e-2,
               mu_d0_params=None, mu_d1_params=None,
               pi_d0_params=None, pi_d1_params=None, 
               p_d0_params=None, p_d1_params=None):
    
    ml_mu_d1 = clone(learner_mu)
    ml_mu_d0 = clone(learner_mu)
    ml_pi_d1 = clone(learner_pi)
    ml_pi_d0 = clone(learner_pi)
    ml_p_d1 = clone(learner_p)
    ml_p_d0 = clone(learner_p)

    if z is None:
        dx = np.column_stack((d, x))
    else:
        dx = np.column_stack((d, x, z))

    if score == 'mar':
        pi_hat_d1 = fit_predict_proba(s, dx, ml_pi_d1, pi_d1_params, smpls, trimming_threshold=trimming_threshold)
        pi_hat_d0 = fit_predict_proba(s, dx, ml_pi_d0, pi_d0_params, smpls, trimming_threshold=trimming_threshold)

        p_hat_d1 = fit_predict_proba(d, x, ml_p_d1, p_d1_params, smpls)
        p_hat_d0_prelim = fit_predict_proba(d, x, ml_p_d0, p_d0_params, smpls)
        p_hat_d0 = [1 - p for p in p_hat_d0_prelim]

        train_cond_d1_s1 = np.intersect1d(np.where(d == 1)[0], np.where(s == 1)[0])
        mu_hat_d1 = fit_predict(y, x, ml_mu_d1, mu_d1_params, smpls, train_cond=train_cond_d1_s1)

        train_cond_d0_s1 = np.intersect1d(np.where(d == 0)[0], np.where(s == 1)[0])
        mu_hat_d0 = fit_predict(y, x, ml_mu_d0, mu_d0_params, smpls, train_cond=train_cond_d0_s1)
    else:
        # create strata for splitting
        strata = d.reshape(-1, 1) + 2 * s.reshape(-1, 1)

        for i_fold, _ in enumerate(smpls):
            ml_mu_d1 = clone(learner_mu)
            ml_mu_d0 = clone(learner_mu)
            ml_pi_d1 = clone(learner_pi)
            ml_pi_d0 = clone(learner_pi)
            ml_p_d1 = clone(learner_p)
            ml_p_d0 = clone(learner_p)

            # set the params for the nuisance learners
            if mu_d1_params is not None:
                ml_mu_d1.set_params(**mu_d1_params[i_fold])
            if mu_d0_params is not None:
                ml_mu_d0.set_params(**mu_d0_params[i_fold])
            if pi_d1_params is not None:
                ml_pi_d1.set_params(**pi_d1_params[i_fold])
            if pi_d0_params is not None:
                ml_pi_d0.set_params(**pi_d0_params[i_fold])
            if p_d1_params is not None:
                ml_p_d1.set_params(**p_d1_params[i_fold])
            if p_d0_params is not None:
                ml_p_d0.set_params(**p_d0_params[i_fold])

            train_inds = smpls[i_fold][0]
            test_inds = smpls[i_fold][1]

            # start nested crossfitting
            train_inds_1, train_inds_2 = train_test_split(train_inds, test_size=0.5,
                                                      random_state=42, stratify=d[train_inds])

    #p_hat_d0 = _trimm(np.array(p_hat_d0), trimming_rule, trimming_threshold)[0]
    #p_hat_d1 = _trimm(np.array(p_hat_d1), trimming_rule, trimming_threshold)[0]

    return mu_hat_d1, mu_hat_d0, pi_hat_d1, pi_hat_d0, p_hat_d1, p_hat_d0  # lists of 3 numpy arrays


def compute_selection(y, mu_hat_d1_list, mu_hat_d0_list, pi_hat_d1_list, pi_hat_d0_list, \
            p_hat_d1_list, p_hat_d0_list, smpls):
    mu_hat_d1 = np.full_like(y, np.nan, dtype='float64')
    mu_hat_d0 = np.full_like(y, np.nan, dtype='float64')
    pi_hat_d1 = np.full_like(y, np.nan, dtype='float64')
    pi_hat_d0 = np.full_like(y, np.nan, dtype='float64')
    p_hat_d1 = np.full_like(y, np.nan, dtype='float64')
    p_hat_d0 = np.full_like(y, np.nan, dtype='float64')
    
    for idx, (_, test_index) in enumerate(smpls):
        mu_hat_d1[test_index] = mu_hat_d1_list[idx]
        mu_hat_d0[test_index] = mu_hat_d0_list[idx]
        pi_hat_d1[test_index] = pi_hat_d1_list[idx]
        pi_hat_d0[test_index] = pi_hat_d0_list[idx]
        p_hat_d1[test_index] = p_hat_d1_list[idx]
        p_hat_d0[test_index] = p_hat_d0_list[idx]

    return mu_hat_d1, mu_hat_d0, pi_hat_d1, pi_hat_d0, p_hat_d1, p_hat_d0


def selection_score_elements(dtreat, dcontrol, mu_d1, mu_d0, 
                        pi_d1, pi_d0, p_d1, p_d0, s, y, normalize_ipw, smpls):
        # psi_a
        psi_a = -1 * np.ones_like(y)
        
        # psi_b
        if normalize_ipw:
            weight_treat = sum(dtreat) / sum((dtreat * s) / (pi_d1 * p_d1))
            weight_control = sum(dcontrol) / sum((dcontrol * s) / (pi_d0 * p_d0))
            
            psi_b1 = weight_treat * ((dtreat * s * (y - mu_d1)) / (p_d1 * pi_d1)) + mu_d1
            psi_b0 = weight_control * ((dcontrol * s * (y - mu_d0)) / (p_d0 * pi_d0)) + mu_d0
        
        else:
            psi_b1 = (dtreat * s * (y - mu_d1)) / (p_d1 * pi_d1) + mu_d1
            psi_b0 = (dcontrol * s * (y - mu_d0)) / (p_d0 * pi_d0) + mu_d0

        psi_b = psi_b1 - psi_b0

        return psi_a, psi_b


def selection_dml1(psi_a, psi_b, smpls):
    thetas = np.zeros(len(smpls))
    n_obs = len(psi_a)

    for idx, (_, test_index) in enumerate(smpls):
        thetas[idx] = - np.mean(psi_b[test_index]) / np.mean(psi_a[test_index])
    theta_hat = np.mean(thetas)

    if len(smpls) > 1:
        se = np.sqrt(var_selection(theta_hat, psi_a, psi_b, n_obs))
    else:
        assert len(smpls) == 1
        test_index = smpls[0][1]
        n_obs = len(test_index)
        se = np.sqrt(var_selection(theta_hat, psi_a[test_index], psi_b[test_index], n_obs))

    return theta_hat, se


def selection_dml2(psi_a, psi_b):
    n_obs = len(psi_a)
    theta_hat = - np.mean(psi_b) / np.mean(psi_a)
    se = np.sqrt(var_selection(theta_hat, psi_a, psi_b, n_obs))

    return theta_hat, se



def var_selection(theta, psi_a, psi_b, n_obs):
    J = np.mean(psi_a)
    var = 1/n_obs * np.mean(np.power(np.multiply(psi_a, theta) + psi_b, 2)) / np.power(J, 2)
    return var