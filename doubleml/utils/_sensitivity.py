import warnings

import numpy as np


def _validate_nu2(nu2, psi_nu2, riesz_rep):
    if np.any(nu2 <= 0):
        warnings.warn(
            "The estimated nu2 is not positive. Re-estimation based on riesz representer (non-orthogonal).",
            UserWarning,
        )
        nu2 = np.mean(np.power(riesz_rep, 2), axis=0, keepdims=True)
        psi_nu2 = np.power(riesz_rep, 2)

    return nu2, psi_nu2
