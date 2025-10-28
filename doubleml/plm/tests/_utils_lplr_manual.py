import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from ...tests._utils import fit_predict, fit_predict_proba, tune_grid_search
from ...utils._estimation import _predict_zero_one_propensity
from ...utils._propensity_score import _trimm


def fit_selection(
        y,
        x,
        d,
        learner_M,
        learner_t,
        learner_m,
        all_smpls,
        score,
        trimming_rule="truncate",
        trimming_threshold=1e-2,
        n_rep=1,
        M_params=None,
        t_params=None,
        m_params=None,
):
    n_obs = len(y)

    thetas = np.zeros(n_rep)
    ses = np.zeros(n_rep)

    all_M_hat = list()
    all_t_hat = list()
    all_m_hat = list()

    for i_rep in range(n_rep):
        smpls = all_smpls[i_rep]

        M_hat_list, t_hat_list, m_hat_list = fit_nuisance_selection(
            y,
            x,
            d,
            learner_M,
            learner_t,
            learner_m,
            smpls,
            score,
            trimming_rule=trimming_rule,
            trimming_threshold=trimming_threshold,
            M_params=M_params,
            t_params=t_params,
            m_params=m_params,
        )

        all_M_hat.append(M_hat)
        all_t_hat.append(t_hat)
        all_m_hat.append(m_hat)

        thetas[i_rep], ses[i_rep] = solve_score(M_hat_list, t_hat_list, m_hat_list)

    theta = np.median(thetas)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(thetas - theta, 2)) / n_obs)

    res = {
        "theta": theta,
        "se": se,
        "thetas": thetas,
        "ses": ses,
        "all_M_hat": all_M_hat,
        "all_t_hat": all_t_hat,
        "all_m_hat": all_m_hat,
        "all_psi_a": all_psi_a,
        "all_psi_b": all_psi_b,
    }

    return res

def solve_score(M_hat, t_hat, m_hat):
    pass

def fit_nuisance_selection(
        y,
        x,
        d,
        learner_M,
        learner_t,
        learner_m,
        smpls,
        score,
        trimming_rule="truncate",
        trimming_threshold=1e-2,
        M_params=None,
        t_params=None,
        m_params=None,
):
    # TODO: complete for lplr
    n_obs = len(y)
    ml_M = clone(learner_M)
    ml_t = clone(learner_t)
    ml_m = clone(learner_m)

    dx = np.column_stack((d, x))

    # initialize empty lists
    g_hat_d1_list = []
    g_hat_d0_list = []
    pi_hat_list = []
    m_hat_list = []

    # create strata for splitting
    strata = d.reshape(-1, 1) + 2 * s.reshape(-1, 1)

    # POTENTIAL OUTCOME Y(1)
    for i_fold, _ in enumerate(smpls):
        ml_g_d1 = clone(learner_g)
        ml_pi = clone(learner_pi)
        ml_m = clone(learner_m)

        # set the params for the nuisance learners
        if g_d1_params is not None:
            ml_g_d1.set_params(**g_d1_params[i_fold])
        if g_d0_params is not None:
            ml_g_d0.set_params(**g_d0_params[i_fold])
        if pi_params is not None:
            ml_pi.set_params(**pi_params[i_fold])
        if m_params is not None:
            ml_m.set_params(**m_params[i_fold])

        train_inds = smpls[i_fold][0]
        test_inds = smpls[i_fold][1]

        # start nested crossfitting
        train_inds_1, train_inds_2 = train_test_split(
            train_inds, test_size=0.5, random_state=42, stratify=strata[train_inds]
        )

        s_train_1 = s[train_inds_1]
        dx_train_1 = dx[train_inds_1, :]

        # preliminary propensity score for selection
        ml_pi_prelim = clone(ml_pi)
        # fit on first part of training set
        ml_pi_prelim.fit(dx_train_1, s_train_1)
        pi_hat_prelim = _predict_zero_one_propensity(ml_pi_prelim, dx)

        # predictions for small pi in denominator
        pi_hat = pi_hat_prelim[test_inds]

        # add selection indicator to covariates
        xpi = np.column_stack((x, pi_hat_prelim))

        # estimate propensity score p using the second training sample
        xpi_train_2 = xpi[train_inds_2, :]
        d_train_2 = d[train_inds_2]
        xpi_test = xpi[test_inds, :]

        ml_m.fit(xpi_train_2, d_train_2)

        m_hat = _predict_zero_one_propensity(ml_m, xpi_test)

        # estimate conditional outcome on second training sample -- treatment
        s1_d1_train_2_indices = np.intersect1d(np.where(d == 1)[0], np.intersect1d(np.where(s == 1)[0], train_inds_2))
        xpi_s1_d1_train_2 = xpi[s1_d1_train_2_indices, :]
        y_s1_d1_train_2 = y[s1_d1_train_2_indices]

        ml_g_d1.fit(xpi_s1_d1_train_2, y_s1_d1_train_2)

        # predict conditional outcome
        g_hat_d1 = ml_g_d1.predict(xpi_test)

        # estimate conditional outcome on second training sample -- control
        s1_d0_train_2_indices = np.intersect1d(np.where(d == 0)[0], np.intersect1d(np.where(s == 1)[0], train_inds_2))
        xpi_s1_d0_train_2 = xpi[s1_d0_train_2_indices, :]
        y_s1_d0_train_2 = y[s1_d0_train_2_indices]

        ml_g_d0.fit(xpi_s1_d0_train_2, y_s1_d0_train_2)

        # predict conditional outcome
        g_hat_d0 = ml_g_d0.predict(xpi_test)

        m_hat = _trimm(m_hat, trimming_rule, trimming_threshold)

        # append predictions on test sample to final list of predictions
        g_hat_d1_list.append(g_hat_d1)
        g_hat_d0_list.append(g_hat_d0)
        pi_hat_list.append(pi_hat)
        m_hat_list.append(m_hat)



    m_hat = np.full_like(y, np.nan, dtype="float64")
    for idx, (_, test_index) in enumerate(smpls):
        M_hat[test_index] = M_hat_list[idx]
        t_hat[test_index] = t_hat_list[idx]
        m_hat[test_index] = m_hat_list[idx]
    return M_hat, t_hat, m_hat


def var_selection(theta, psi_a, psi_b, n_obs):
    J = np.mean(psi_a)
    var = 1 / n_obs * np.mean(np.power(np.multiply(psi_a, theta) + psi_b, 2)) / np.power(J, 2)
    return var


def tune_nuisance(y, x, d, ml_M, ml_t, ml_m, smpls, n_folds_tune, param_grid_M, param_grid_t, param_grid_m):
    dx = np.column_stack((x, d))

    M_tune_res = tune_grid_search(y, dx, ml_M, smpls, param_grid_M, n_folds_tune)

    m_tune_res = tune_grid_search(d, x, ml_m, smpls, param_grid_m, n_folds_tune)

    t_tune_res = tune_grid_search(d, x, ml_t, smpls, param_grid_t, n_folds_tune)

    M_best_params = [xx.best_params_ for xx in M_tune_res]
    t_best_params = [xx.best_params_ for xx in t_tune_res]
    m_best_params = [xx.best_params_ for xx in m_tune_res]

    t_tune_res = tune_grid_search(t_targets, x, ml_t, smpls, param_grid_t, n_folds_tune)
