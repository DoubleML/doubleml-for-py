import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import RegressionResults

import doubleml as dml
from doubleml.datasets import fetch_401K

from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV
import statsmodels.api as sm

from scipy.stats import norm
from scipy.linalg import sqrtm

import patsy

import matplotlib.pyplot as plt


def polynomial_fit(X: np.array, y: np.array, max_degree: int) -> tuple[RegressionResults, int]:
    """
    Polynomial regression of y with respect to X. CV is implemented to test polynomials from degree 1 up to max_degree.
    Only the degree with minimal residuals is returned.

    X in this case should be the feature for which to calculate the CATE and y the robust score described in
    Semenova 2.2

    Parameters
    ----------
    X: features of the regression (variable for which we want to calculate the CATE)
    y: target variable (robust score)
    max_degree: maximum degree of the polynomials

    Returns
    -------
    fitted model and degree of the polynomials
    """
    # todo: assert correct dimensions of the parameters
    # todo: is it really important that the polynomials be orthogonal?
    # todo: allow for multiple variables in X
    # First we find which is the degree with minimal error
    cv_errors = np.zeros(max_degree)
    for degree in range(1, max_degree + 1):
        pf = PolynomialFeatures(degree=degree, include_bias=True)
        X_poly = pf.fit_transform(X)
        model = sm.OLS(y, X_poly)
        results = model.fit()
        influence = results.get_influence()
        leverage = influence.hat_matrix_diag  # this is what semenova uses (leverage)
        cv_errors[degree - 1] = np.sum((results.resid / (1 - leverage)) ** 2)

    # Degree chosen by cross-validation (we add one because degree zero is not included)
    cv_degree = np.argmin(cv_errors) + 1

    # Estimate coefficients
    pf = PolynomialFeatures(degree=cv_degree, include_bias=True)
    x_poly = pf.fit_transform(X)
    model = sm.OLS(y, x_poly)
    results = model.fit()
    return results, cv_degree


def splines_fit(X: np.array, y: np.array, max_knots: int, degree: int = 3) -> tuple[RegressionResults, int]:
    """
    Splines interpolation of y with respect to X. CV is implemented to test splines of number of knots
    from knots 2 up to max_knots. Only the knots with minimal residuals is returned.

    X in this case should be the feature for which to calculate the CATE and y the robust score described in
    Semenova 2.2

    Parameters
    ----------
    X: features of the regression (variable for which we want to calculate the CATE)
    y: target variable (robust score)
    max_knots: maximum number of knots in the splines interpolation
    degree: degree of the splines polynomials to be used

    Returns
    -------
    fitted model and degree of the polynomials
    """
    # todo: assert correct dimensions of the parameters
    # todo: is it really important that the polynomials be orthogonal?
    # todo: allow for multiple variables in X
    # First we find which is the n_knots with minimal error
    cv_errors = np.zeros(max_knots)
    for n_knots in range(2, max_knots + 1):
        breaks = np.quantile(X, q=np.array(range(0, n_knots + 1)) / n_knots)
        X_splines = patsy.bs(X, knots=breaks[:-1], degree=degree)
        model = sm.OLS(y, X_splines)
        results = model.fit()
        influence = results.get_influence()
        leverage = influence.hat_matrix_diag  # this is what semenova uses (leverage)
        cv_errors[n_knots - 2] = np.sum((results.resid / (1 - leverage)) ** 2)

    # Degree chosen by cross-validation (we add one because degree zero is not included)
    cv_knots = np.argmin(cv_errors) + 1

    # Estimate coefficients
    breaks = np.quantile(X, q=np.array(range(0, cv_knots + 1)) / cv_knots)
    X_splines = patsy.bs(X, knots=breaks[:-1], degree=degree)
    model = sm.OLS(y, X_splines)
    results = model.fit()
    return results, cv_knots


def calculate_bootstrap_tstat(regressors_grid: np.array, omega_hat: np.array, alpha: float, n_samples_bootstrap: int) \
        -> float:
    """
    This function calculates the critical value of the confidence bands of the bootstrapped t-statistics
    from def. 2.7 in Semenova.

    Parameters
    ----------
    regressors_grid: support of the variable of interested for which to calculate the t-statistics
    omega_hat: covariance matrix
    alpha: p-value
    n_samples_bootstrap: number of samples to generate for the normal distribution draw

    Returns
    -------
    float with the critical value of the t-statistic
    """
    # don't need sqrt(N) because it cancels out with the numerator
    numerator_grid = regressors_grid @ sqrtm(omega_hat)
    # we take the diagonal because in the paper the multiplication is p(x)'*Omega*p(x),
    # where p(x) is the vector of basis functions
    denominator_grid = np.sqrt(np.diag(regressors_grid @ omega_hat @ np.transpose(regressors_grid)))

    norm_numerator_grid = numerator_grid.copy()
    for k in range(numerator_grid.shape[0]):
        norm_numerator_grid[k, :] = numerator_grid[k, :] / denominator_grid[k]

    t_maxs = np.amax(np.abs(norm_numerator_grid @ np.random.normal(size=numerator_grid.shape[1] * n_samples_bootstrap)
                            .reshape(numerator_grid.shape[1], n_samples_bootstrap)), axis=1)
    return np.quantile(t_maxs, q=1 - alpha)


def second_stage_estimation(X: np.array, y: np.array, method: str, max_degree: int, max_knots: int, degree: int,
                            alpha: float, n_nodes: int, n_samples_bootstrap: int) -> dict:
    """
    Calculates the CATE with respect to variable X by polynomial or splines approximation.
    It calculates it on an equidistant grid of n_nodes points over the range of X

    Parameters
    ----------
    X: variable whose CATE is calculated
    y: robust dependent variable, as defined in 2.2 in Semenova
    method: "poly" or "splines", chooses which method to approximate with
    max_degree: maximum degree of the polynomial approximation
    max_knots: max knots for the splines approximation
    degree: degree of the polynomials used in the splines
    alpha: p-value for the confidence intervals
    n_nodes: number of points of X for which we wish to calculate the CATE
    n_samples_bootstrap: how many samples to use to calculate the t-statistics described in 2.6

    Returns
    -------
    A dictionary containing the estimated CATE (g_hat), with upper and lower confidence bounds (both simultaneous as
    well as pointwise), fitted linear model and grid of X for which the CATE was calculated

    """
    x_grid = np.linspace(np.min(X), np.max(X), n_nodes).reshape(-1, 1)

    if method == "poly":
        fitted_model, poly_degree = polynomial_fit(X=X, y=y, max_degree=max_degree)

        # Build the set of datapoints in X that will be used for prediction
        pf = PolynomialFeatures(degree=poly_degree, include_bias=True)
        regressors_grid = pf.fit_transform(x_grid)

    elif method == "splines":
        fitted_model, n_knots = splines_fit(X=X, y=y, degree=degree, max_knots=max_knots)

        # Build the set of datapoints in X that will be used for prediction
        breaks = np.quantile(X, q=np.array(range(0, n_knots + 1)) / n_knots)
        regressors_grid = patsy.bs(x_grid, knots=breaks[:-1], degree=degree)

    else:
        raise NotImplementedError("The specified method is not implemented. Please use 'poly' or 'splines'")

    g_hat = regressors_grid @ fitted_model.params
    # we can get the HCO matrix directly from the model object
    hcv_coeff = fitted_model.cov_HC0
    standard_error = np.sqrt(np.diag(regressors_grid @ hcv_coeff @ np.transpose(regressors_grid)))
    # Lower pointwise CI
    g_hat_lower_point = g_hat + norm.ppf(q=alpha / 2) * standard_error
    # Upper pointwise CI
    g_hat_upper_point = g_hat + norm.ppf(q=1 - alpha / 2) * standard_error

    max_t_stat = calculate_bootstrap_tstat(regressors_grid=regressors_grid,
                                           omega_hat=hcv_coeff,
                                           alpha=alpha,
                                           n_samples_bootstrap=n_samples_bootstrap)
    # Lower simultaneous CI
    g_hat_lower = g_hat - max_t_stat * standard_error
    # Upper simultaneous CI
    g_hat_upper = g_hat + max_t_stat * standard_error
    results_dict = {
        "g_hat": g_hat,
        "g_hat_lower": g_hat_lower,
        "g_hat_upper": g_hat_upper,
        "g_hat_lower_point": g_hat_lower_point,
        "g_hat_upper_point": g_hat_upper_point,
        "x_grid": x_grid,
        "fitted_model": fitted_model
    }
    return results_dict


# Reproducing the same results as in https://github.com/microsoft/EconML/blob/master/notebooks/Double%20Machine%20Learning%20Examples.ipynb

# Treatment effect function
def transform_te(x):
    return np.exp(4 + 2*x[0])
    # return 3


# DGP constants
np.random.seed(123)


def create_synthetic_data(n: int, n_w: int, support_size: int, n_x: int):
    """
    Creates a synthetic example based on example 2 of https://github.com/microsoft/EconML/blob/master/notebooks/Double%20Machine%20Learning%20Examples.ipynb

    Parameters
    ----------
    n_samples
    n_w
    support_size
    n_x

    Returns
    -------

    """
    # Outcome support
    # With the next two lines we are effectively choosing the matrix gamma in the example
    support_Y = np.random.choice(np.arange(n_w), size=support_size, replace=False)
    coefs_Y = np.random.uniform(0, 1, size=support_size)
    # Define the function to generate the noise
    epsilon_sample = lambda n: np.random.uniform(-1, 1, size=n)
    # Treatment support
    # Assuming the matrices gamma and beta have the same non-zero components
    support_T = support_Y
    coefs_T = np.random.uniform(0, 1, size=support_size)
    # Define the function to generate the noise
    eta_sample = lambda n: np.random.uniform(-1, 1, size=n)

    # Generate controls, covariates, treatments and outcomes
    W = np.random.normal(0, 1, size=(n, n_w))
    X = np.random.uniform(0, 1, size=(n, n_x))
    # Heterogeneous treatment effects
    TE = np.array([transform_te(x_i) for x_i in X])
    # Define treatment
    log_odds = np.dot(W[:, support_T], coefs_T) + eta_sample(n)
    T_sigmoid = 1 / (1 + np.exp(-log_odds))
    T = np.array([np.random.binomial(1, p) for p in T_sigmoid])
    # Define the outcome
    Y = TE * T + np.dot(W[:, support_Y], coefs_Y) + epsilon_sample(n)

    # Now we build the dataset
    y_df = pd.DataFrame({'y': Y})
    x_df = pd.DataFrame({'x': X.reshape(-1)})
    t_df = pd.DataFrame({'t': T})
    w_df = pd.DataFrame(data=W, index=np.arange(W.shape[0]), columns=[f'w_{i}' for i in range(W.shape[1])])

    data = pd.concat([y_df, x_df, t_df, w_df], axis=1)

    covariates = list(w_df.columns.values) + list(x_df.columns.values)
    return data, covariates


def plot_results(dict_results: dict, n, n_w, support_size, n_x, method):
    df_plot = pd.DataFrame({'g_hat': dict_results["g_hat"],
                            "g_hat_lower": dict_results["g_hat_lower"],
                            "g_hat_upper": dict_results["g_hat_upper"],
                            "g_hat_lower_point": dict_results["g_hat_lower_point"],
                            "g_hat_upper_point": dict_results["g_hat_upper_point"],
                            "x_grid": dict_results["x_grid"].flatten(),
                            # 'True_g': 3
                            'True_g': [transform_te(x) for x in dict_results["x_grid"]]
                            })
    plt.clf()
    plt.plot(df_plot["x_grid"], df_plot["g_hat"], "k", linestyle="solid", label="mean effect")
    plt.plot(df_plot["x_grid"], df_plot["g_hat_lower"], "b", linestyle="dotted", label="simultaneous lower")
    plt.plot(df_plot["x_grid"], df_plot["g_hat_upper"], "b", linestyle="dotted", label="simultaneous higher")
    plt.plot(df_plot["x_grid"], df_plot["g_hat_lower_point"], "darkgreen", linestyle="dashed", label="pointwise lower")
    plt.plot(df_plot["x_grid"], df_plot["g_hat_upper_point"], "darkgreen", linestyle="dashed", label="pointwise upper")
    plt.plot(df_plot["x_grid"], df_plot["True_g"], "red", linestyle="solid", label="True effect")
    plt.legend(loc="upper left")
    plt.title(f"CATE with {method}, n={n}, n_w={n_w}, sup_size={support_size}, n_x={n_x},")
    plt.savefig(
        rf"\\ad.uni-hamburg.de\redir\redir0101\BBB1675\Documents\CATE\figures\CATE_{method}_n{n}_nw{n_w}_sup{support_size}_nx{n_x}.png")


n = 2000
n_w = 30
support_size = 5
n_x = 1

for n in [2000, 3000, 4000]:
    for n_w in [5, 10, 20, 30]:
        for support_size in [5, 10, 20, 30]:
            if support_size <= n_w:
                for method in ["poly", "splines"]:
                    print(f"CATE with {method}, n={n}, n_w={n_w}, sup_size={support_size}, n_x={n_x},")

                    # Create data
                    data, covariates = create_synthetic_data(n=n, n_w=n_w, support_size=support_size, n_x=n_x)
                    data_dml_base = dml.DoubleMLData(data,
                                                     y_col='y',
                                                     d_cols='t',
                                                     x_cols=covariates)

                    # First stage estimation
                    # Lasso regression
                    lasso_reg = LassoCV()
                    randomForest_class = RandomForestClassifier(n_estimators=500)

                    np.random.seed(123)

                    dml_irm = dml.DoubleMLIRM(data_dml_base,
                                              ml_g=lasso_reg,
                                              ml_m=randomForest_class,
                                              trimming_threshold=0.01,
                                              n_folds=5
                                              )
                    print("Training first stage")
                    dml_irm.fit(store_predictions=True)

                    # Psi_b is the part of the score that we are interested in (It is the robust Y in Semenova)
                    y_robust = dml_irm.psi_b  # Q: why is y with dim 3?
                    # We reshape it so that it is a nx1 vector
                    y_robust = y_robust.reshape(-1, 1)

                    X = np.array(data['x'])
                    X = X.reshape(-1, 1)
                    max_degree = 3
                    degree = 3
                    max_knots = 20
                    alpha = 0.05
                    n_nodes = 90
                    n_samples_bootstrap = 10000
                    print("Training second stage")
                    dict_results = second_stage_estimation(X=X, y=y_robust, method=method,
                                                           max_degree=max_degree,
                                                           degree=degree,
                                                           max_knots=max_knots,
                                                           alpha=alpha,
                                                           n_nodes=n_nodes,
                                                           n_samples_bootstrap=n_samples_bootstrap)
                    print("Saving results")
                    plot_results(dict_results, n, n_w=n_w, support_size=support_size, n_x=n_x, method=method)
