import numpy as np
from sklearn.base import clone

from ..pq import DoubleMLPQ
from ...double_ml_data import DoubleMLData

from ...tests._utils_boot import draw_weights
from ...utils._estimation import _default_kde


def fit_qte(y, x, d, quantiles, learner_g, learner_m, all_smpls, n_rep=1,
            trimming_rule='truncate', trimming_threshold=1e-2, kde=_default_kde,
            normalize_ipw=True, draw_sample_splitting=True):

    n_obs = len(y)
    n_quantiles = len(quantiles)
    n_folds = len(all_smpls[0])

    dml_data = DoubleMLData.from_arrays(x, y, d)

    qtes = np.zeros((n_quantiles, n_rep))
    ses = np.zeros((n_quantiles, n_rep))

    scaled_scores = np.zeros((n_obs, n_quantiles, n_rep))

    for i_quant in range(n_quantiles):
        # initialize models for both potential quantiles
        model_PQ_0 = DoubleMLPQ(dml_data,
                                clone(learner_g),
                                clone(learner_m),
                                quantile=quantiles[i_quant],
                                treatment=0,
                                n_folds=n_folds,
                                n_rep=n_rep,
                                trimming_rule=trimming_rule,
                                trimming_threshold=trimming_threshold,
                                kde=kde,
                                normalize_ipw=normalize_ipw,
                                draw_sample_splitting=False)
        model_PQ_1 = DoubleMLPQ(dml_data,
                                clone(learner_g),
                                clone(learner_m),
                                quantile=quantiles[i_quant],
                                treatment=1,
                                n_folds=n_folds,
                                n_rep=n_rep,
                                trimming_rule=trimming_rule,
                                trimming_threshold=trimming_threshold,
                                kde=kde,
                                normalize_ipw=normalize_ipw,
                                draw_sample_splitting=False)

        # synchronize the sample splitting
        model_PQ_0.set_sample_splitting(all_smpls)
        model_PQ_1.set_sample_splitting(all_smpls)

        model_PQ_0.fit()
        model_PQ_1.fit()

        # Quantile Treatment Effects
        qtes[i_quant, :] = model_PQ_1.all_coef - model_PQ_0.all_coef

        for i_rep in range(n_rep):
            J0 = model_PQ_0.psi_deriv[:, i_rep, 0].mean()
            J1 = model_PQ_1.psi_deriv[:, i_rep, 0].mean()
            score0 = model_PQ_0.psi[:, i_rep, 0]
            score1 = model_PQ_1.psi[:, i_rep, 0]
            omega = score1 / J1 - score0 / J0

            scaled_scores[:, i_quant, i_rep] = omega

            var_scaling_factor = n_obs
            var = 1 / var_scaling_factor * np.mean(np.power(omega, 2))
            ses[i_quant, i_rep] = np.sqrt(var)

    qte = np.median(qtes, 1)
    se = np.zeros(n_quantiles)
    for i_quant in range(n_quantiles):
        se[i_quant] = np.sqrt(np.median(np.power(ses[i_quant, :], 2) * n_obs +
                                        np.power(qtes[i_quant, :] - qte[i_quant], 2)) / n_obs)

    res = {'qte': qte, 'se': se,
           'qtes': qtes, 'ses': ses,
           'scaled_scores': scaled_scores}

    return res


def boot_qte(scaled_scores, ses, quantiles, all_smpls, n_rep, bootstrap, n_rep_boot):
    n_quantiles = len(quantiles)
    boot_t_stat = np.zeros((n_rep_boot, n_quantiles, n_rep))
    for i_rep in range(n_rep):
        n_obs = scaled_scores.shape[0]
        weights = draw_weights(bootstrap, n_rep_boot, n_obs)
        for i_quant in range(n_quantiles):
            boot_t_stat[:, i_quant, i_rep] = np.matmul(weights, scaled_scores[:, i_quant, i_rep]) / \
                (n_obs * ses[i_quant, i_rep])

    return boot_t_stat
