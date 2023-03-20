import numpy as np
from sklearn.base import clone
import pandas as pd
from scipy.stats import norm

from ..double_ml_pq import DoubleMLPQ
from ..double_ml_data import DoubleMLData

from ._utils_boot import draw_weights
from .._utils import _default_kde


def fit_qte(y, x, d, quantiles, learner_g, learner_m, all_smpls, n_rep=1, dml_procedure='dml2',
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
                                dml_procedure=dml_procedure,
                                trimming_rule=trimming_rule,
                                trimming_threshold=trimming_threshold,
                                kde=kde,
                                normalize_ipw=normalize_ipw,
                                draw_sample_splitting=False,
                                apply_cross_fitting=True)
        model_PQ_1 = DoubleMLPQ(dml_data,
                                clone(learner_g),
                                clone(learner_m),
                                quantile=quantiles[i_quant],
                                treatment=1,
                                n_folds=n_folds,
                                n_rep=n_rep,
                                dml_procedure=dml_procedure,
                                trimming_rule=trimming_rule,
                                trimming_threshold=trimming_threshold,
                                kde=kde,
                                normalize_ipw=normalize_ipw,
                                draw_sample_splitting=False,
                                apply_cross_fitting=True)

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


def boot_qte(scaled_scores, ses, quantiles, all_smpls, n_rep, bootstrap, n_rep_boot, apply_cross_fitting):
    n_quantiles = len(quantiles)
    boot_qte = np.zeros((n_quantiles, n_rep_boot * n_rep))
    boot_t_stat = np.zeros((n_quantiles, n_rep_boot * n_rep))
    for i_rep in range(n_rep):
        n_obs = scaled_scores.shape[0]
        weights = draw_weights(bootstrap, n_rep_boot, n_obs)
        for i_quant in range(n_quantiles):
            i_start = i_rep * n_rep_boot
            i_end = (i_rep + 1) * n_rep_boot

            boot_qte[i_quant, i_start:i_end] = np.matmul(weights, scaled_scores[:, i_quant, i_rep]) / n_obs
            boot_t_stat[i_quant, i_start:i_end] = np.matmul(weights, scaled_scores[:, i_quant, i_rep]) / \
                (n_obs * ses[i_quant, i_rep])

    return boot_qte, boot_t_stat


def confint_qte(coef, se, quantiles, boot_t_stat=None, joint=True, level=0.95):
    a = (1 - level)
    ab = np.array([a / 2, 1. - a / 2])
    if joint:
        sim = np.amax(np.abs(boot_t_stat), 0)
        hatc = np.quantile(sim, 1 - a)
        ci = np.vstack((coef - se * hatc, coef + se * hatc)).T
    else:
        fac = norm.ppf(ab)
        ci = np.vstack((coef + se * fac[0], coef + se * fac[1])).T

    df_ci = pd.DataFrame(ci,
                         columns=['{:.1f} %'.format(i * 100) for i in ab],
                         index=quantiles)
    return df_ci
