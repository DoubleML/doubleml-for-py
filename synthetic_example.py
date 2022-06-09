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


# Reproducing the same results as in https://github.com/microsoft/EconML/blob/master/notebooks/Double%20Machine%20Learning%20Examples.ipynb

# Treatment effect function
def transform_te(x):
    return np.exp(4 + 2 * x[0])
    # return 3


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
    plt.savefig(rf".\CATE_{method}_n{n}_nw{n_w}_sup{support_size}_nx{n_x}.png")

# DGP constants
np.random.seed(123)
n = 2000
n_w = 30
support_size = 5
n_x = 1

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

dict_results = dml_irm.cate(cate_var="x",
                            method="splines",
                            alpha=0.05,
                            n_grid_nodes=90,
                            n_samples_bootstrap=10000,
                            cv=False,
                            splines_knots=30,
                            splines_degree=3)

plot_results(dict_results, n, n_w=n_w, support_size=support_size, n_x=n_x, method="splines")
