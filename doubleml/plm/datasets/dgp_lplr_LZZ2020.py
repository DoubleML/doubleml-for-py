import numpy as np
import pandas as pd
from scipy.special import expit

from doubleml.data import DoubleMLData
from doubleml.utils._aliases import _get_array_alias, _get_data_frame_alias, _get_dml_data_alias

_array_alias = _get_array_alias()
_data_frame_alias = _get_data_frame_alias()
_dml_data_alias = _get_dml_data_alias()


def make_lplr_LZZ2020(n_obs=500, dim_x=20, alpha=0.5, return_type="DoubleMLData", balanced_r0=True, treatment="continuous"):
    r"""
    Generates synthetic data for a logistic partially linear regression model, as in Liu et al. (2021),
    designed for use in double/debiased machine learning applications.

    The data generating process is defined as follows:

    - Covariates :math:`x_i \sim \mathcal{N}(0, \Sigma)`, where :math:`\Sigma_{kj} = 0.2^{|j-k|}`.
    - Treatment :math:`d_i = a_0(x_i)` (or a binary transformation thereof, depending on the `treatment` parameter).
    - Propensity score :math:`p_i = \sigma(\alpha d_i + r_0(x_i))`, where :math:`\sigma(\cdot)` is the logistic function.
    - Outcome :math:`y_i \sim \text{Bernoulli}(p_i)`.

    The nuisance functions are defined as:

    .. math::
        \begin{aligned}
        a_0(x_i) &= \frac{2}{1 + \exp(x_{i,1})} - \frac{2}{1 + \exp(x_{i,2})} + \sin(x_{i,3}) + \cos(x_{i,4}) \\
                 &\quad + 0.5 \cdot \mathbb{1}(x_{i,5} > 0) - 0.5 \cdot \mathbb{1}(x_{i,6} > 0) + 0.2\, x_{i,7} x_{i,8}
                 - 0.2\, x_{i,9} x_{i,10} \\
        r_0(x_i) &= 0.1\, x_{i,1} x_{i,2} x_{i,3} + 0.1\, x_{i,4} x_{i,5} + 0.1\, x_{i,6}^3 - 0.5 \sin^2(x_{i,7}) \\
                 &\quad + 0.5 \cos(x_{i,8}) + \frac{1}{1 + x_{i,9}^2} - \frac{1}{1 + \exp(x_{i,10})} \\
                 &\quad + 0.25 \cdot \mathbb{1}(x_{i,11} > 0) - 0.25 \cdot \mathbb{1}(x_{i,13} > 0)
        \end{aligned}

    Parameters
    ----------
    n_obs : int, default=500
        Number of observations to simulate.

    dim_x : int, default=20
        Number of covariates.

    alpha : float, default=0.5
        Value of the causal parameter.

    return_type : str, default="DoubleMLData"
        Determines the return format. One of:

        - 'DoubleMLData' or DoubleMLData: returns a ``DoubleMLData`` object.
        - 'DataFrame', 'pd.DataFrame' or pd.DataFrame: returns a ``pandas.DataFrame``.
        - 'array', 'np.ndarray', 'np.array' or np.ndarray: returns tuple of numpy arrays (x, y, d, p).

    balanced_r0 : bool, default=True
        If True, uses the "balanced" r_0 specification (smaller magnitude / more balanced
        heterogeneity). If False, uses an "unbalanced" r_0 specification with larger
        share of Y=0.

    treatment : str, default="continuous"
        Type of treatment variable. One of "continuous", "binary", or "binary_unbalanced".
        Determines how the treatment d is generated from a_0(x):

        - "continuous": d = a_0(x) (continuous treatment).
        - "binary":    d ~ Bernoulli( sigmoid(a_0(x) - mean(a_0(x))) ) .
        - "binary_unbalanced": d ~ Bernoulli( sigmoid(a_0(x)) ).

    Returns
    -------
    Union[DoubleMLData, pd.DataFrame, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        The generated data in the specified format.

    References
    ----------
    Liu, Molei, Yi Zhang, and Doudou Zhou. 2021.
    "Double/Debiased Machine Learning for Logistic Partially Linear Model."
    The Econometrics Journal 24 (3): 559–88.
    doi:`10.1093/ectj/utab019 <https://doi.org/10.1093/ectj/utab019>`_.

    """

    if balanced_r0:

        def r_0(X):
            return (
                0.1 * X[:, 0] * X[:, 1] * X[:, 2]
                + 0.1 * X[:, 3] * X[:, 4]
                + 0.1 * X[:, 5] ** 3
                + -0.5 * np.sin(X[:, 6]) ** 2
                + 0.5 * np.cos(X[:, 7])
                + 1 / (1 + X[:, 8] ** 2)
                + -1 / (1 + np.exp(X[:, 9]))
                + 0.25 * np.where(X[:, 10] > 0, 1, 0)
                + -0.25 * np.where(X[:, 12] > 0, 1, 0)
            )

    else:

        def r_0(X):
            return (
                0.1 * X[:, 0] * X[:, 1] * X[:, 2]
                + 0.1 * X[:, 3] * X[:, 4]
                + 0.1 * X[:, 5] ** 3
                + -0.5 * np.sin(X[:, 6]) ** 2
                + 0.5 * np.cos(X[:, 7])
                + 4 / (1 + X[:, 8] ** 2)
                + -1 / (1 + np.exp(X[:, 9]))
                + 1.5 * np.where(X[:, 10] > 0, 1, 0)
                + -0.25 * np.where(X[:, 12] > 0, 1, 0)
            )

    def a_0(X):
        return (
            2 / (1 + np.exp(X[:, 0]))
            + -2 / (1 + np.exp(X[:, 1]))
            + 1 * np.sin(X[:, 2])
            + 1 * np.cos(X[:, 3])
            + 0.5 * np.where(X[:, 4] > 0, 1, 0)
            + -0.5 * np.where(X[:, 5] > 0, 1, 0)
            + 0.2 * X[:, 6] * X[:, 7]
            + -0.2 * X[:, 8] * X[:, 9]
        )

    sigma = np.full((dim_x, dim_x), 0.2)
    np.fill_diagonal(sigma, 1)

    x = np.random.multivariate_normal(np.zeros(dim_x), sigma, size=n_obs)
    np.clip(x, -2, 2, out=x)

    if treatment == "continuous":
        d = a_0(x)
    elif treatment == "binary":
        d_cont = a_0(x)
        d = np.random.binomial(1, expit(d_cont - d_cont.mean()))
    elif treatment == "binary_unbalanced":
        d_cont = a_0(x)
        d = np.random.binomial(1, expit(d_cont))
    else:
        raise ValueError("Invalid treatment type.")

    p = expit(alpha * d[:] + r_0(x))

    y = np.random.binomial(1, p)

    if return_type in _array_alias:
        return x, y, d, p
    elif return_type in _data_frame_alias + _dml_data_alias:
        x_cols = [f"X{i + 1}" for i in np.arange(dim_x)]
        data = pd.DataFrame(np.column_stack((x, y, d, p)), columns=x_cols + ["y", "d", "p"])
        if return_type in _data_frame_alias:
            return data
        else:
            return DoubleMLData(data, "y", "d", x_cols)
    else:
        raise ValueError("Invalid return_type.")
