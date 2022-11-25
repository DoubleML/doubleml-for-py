import numpy as np
from sklearn.base import clone

from ..double_ml_pq import DoubleMLPQ
from ..double_ml_data import DoubleMLData

def fit_qte(y, x, d, quantiles, learner_g, learner_m, all_smpls, n_rep=1, dml_procedure='dml2',
            trimming_rule='truncate', trimming_threshold=1e-12, h=None,
            normalize=True, draw_sample_splitting=True):

    n_obs = len(y)
    n_quantiles = len(quantiles)
    n_folds = len(all_smpls[0])

    dml_data = DoubleMLData.from_arrays(x, y, d)

    qtes = np.zeros((n_quantiles, n_rep))
    ses = np.zeros((n_quantiles, n_rep))

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
                                h=h,
                                normalize=normalize,
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
                                h=h,
                                normalize=normalize,
                                draw_sample_splitting=False,
                                apply_cross_fitting=True)

        # synchronize the sample splitting
        model_PQ_0.set_sample_splitting(all_smpls)
        model_PQ_1.set_sample_splitting(all_smpls)

        model_PQ_0.fit()
        model_PQ_1.fit()

        # Quantile Treatment Effects
        qtes[i_quant, :] = model_PQ_1.all_coef - model_PQ_0.all_coef

    qte = np.median(qtes, 1)
    se = np.sqrt(np.median(np.power(ses, 2) * n_obs + np.power(qtes - qte, 2)) / n_obs)

    res = {'qte': qte, 'se': se,
           'qtes': qtes, 'ses': ses}

    return res