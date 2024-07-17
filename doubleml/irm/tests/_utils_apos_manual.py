import numpy as np
from sklearn.base import clone

from ..apo import DoubleMLAPO
from ...double_ml_data import DoubleMLData


def fit_apos(y, x, d,
             learner_g, learner_m, treatment_levels, all_smpls, score,
             n_rep=1, trimming_rule='truncate',
             normalize_ipw=False, trimming_threshold=1e-2):
    n_obs = len(y)
    n_treatments = len(treatment_levels)
    n_folds = len(all_smpls[0])

    dml_data = DoubleMLData.from_arrays(x, y, d)

    all_apos = np.zeros((n_treatments, n_rep))
    all_se = np.zeros((n_treatments, n_rep))
    apo_scaled_score = np.zeros((n_obs, n_treatments, n_rep))

    for i_level in range(n_treatments):
        model_APO = DoubleMLAPO(
            dml_data,
            clone(learner_g),
            clone(learner_m),
            treatment_level=treatment_levels[i_level],
            n_folds=n_folds,
            n_rep=n_rep,
            score=score,
            trimming_rule=trimming_rule,
            trimming_threshold=trimming_threshold,
            normalize_ipw=normalize_ipw,
            draw_sample_splitting=False
        )

        # synchronize the sample splitting
        model_APO.set_sample_splitting(all_smpls)
        model_APO.fit()

        all_apos[i_level, :] = model_APO.all_coef
        all_se[i_level, :] = model_APO.all_se

        for i_rep in range(n_rep):
            J = model_APO.psi_deriv[:, i_rep, 0].mean()
            apo_psi = model_APO.psi[:, i_rep, 0]

            apo_scaled_score[:, i_level, i_rep] = apo_psi / J

        apos = np.median(all_apos, axis=1)
        se = np.zeros(n_treatments)
        for i_level in range(n_treatments):
            se[i_level] = np.sqrt(np.median(np.power(all_se[i_level, :], 2) * n_obs +
                                            np.power(all_apos[i_level, :] - all_apos[i_level], 2)) / n_obs)

    res = {'apos': apos, 'se': se,
           'all_apos': all_apos, 'all_se': all_se,
           'apo_scaled_score': apo_scaled_score}
    return res
