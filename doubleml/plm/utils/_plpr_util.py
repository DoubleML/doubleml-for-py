import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


def extend_data(data):
    data = data.copy()
    poly = PolynomialFeatures(2, include_bias=False)

    xdat = data.loc[:,data.columns.str.startswith('x') & ~data.columns.str.contains('lag')]
    xpoly = poly.fit_transform(xdat)
    x_p3 = xdat**3

    x_pol_nam = poly.get_feature_names_out()
    x_cols_p3 = [f'x{i + 1}^3' for i in np.arange(xdat.shape[1])]
    
    if data.columns.str.startswith('m_x').any():
        xmdat = data.loc[:,data.columns.str.startswith('m_x')]
        xmpoly = poly.fit_transform(xmdat)
        xm_p3 = xmdat**3

        xm_pol_nam = poly.get_feature_names_out()
        xm_cols_p3 = [f'm_x{i + 1}^3' for i in np.arange(xmdat.shape[1])]

        X_all = np.column_stack((xpoly, x_p3, xmpoly, xm_p3))
        x_df = pd.DataFrame(X_all, columns = list(x_pol_nam) + x_cols_p3 + list(xm_pol_nam) + xm_cols_p3)
        df_ext = data[['id', 'time', 'd', 'y', 'm_d']].join(x_df)

    elif data.columns.str.contains('_lag').any():
        xldat = data.loc[:,data.columns.str.contains('_lag')]
        xlpoly = poly.fit_transform(xldat)
        xl_p3 = xldat**3

        xl_pol_nam = poly.get_feature_names_out()
        xl_cols_p3 = [f'x{i + 1}_lag^3' for i in np.arange(xldat.shape[1])]

        X_all = np.column_stack((xpoly, x_p3, xlpoly, xl_p3))
        x_df = pd.DataFrame(X_all, columns = list(x_pol_nam) + x_cols_p3 + list(xl_pol_nam) + xl_cols_p3)
        df_ext = data[['id', 'time', 'd_diff', 'y_diff']].join(x_df)

    else:
        X_all = np.column_stack((xpoly, x_p3))
        x_df = pd.DataFrame(X_all, columns = list(x_pol_nam) + x_cols_p3)
        df_ext = data[['id', 'time', 'd', 'y']].join(x_df)

    return df_ext


def cre_fct(data):
    df = data.copy()
    id_means = df.loc[:,~df.columns.isin(['time', 'y'])].groupby(["id"]).transform('mean')
    df = df.join(id_means.rename(columns=lambda x: "m_" + x))
    return df


def fd_fct(data):
    df = data.copy()
    shifted = df.loc[:,~df.columns.isin(['d', 'y', 'time'])].groupby(["id"]).shift(1)
    first_diff = df.loc[:,df.columns.isin(['id', 'd', 'y'])].groupby(["id"]).diff()
    df_fd = df.join(shifted.rename(columns=lambda x: x +"_lag"))
    df_fd = df_fd.join(first_diff.rename(columns=lambda x: x +"_diff"))
    df = df_fd.dropna(subset=['x1_lag']).reset_index(drop=True)
    return df


def wd_fct(data):
    df = data.copy()
    df_demean = df.loc[:,~df.columns.isin(['time'])].groupby(["id"]).transform(lambda x: x - x.mean())
    # add xbar (the grand mean allows a consistent estimate of the constant term)
    within_means = df_demean + df.loc[:,~df.columns.isin(['id','time'])].mean()
    df_wd = df.loc[:,df.columns.isin(['id','time'])]
    df = df_wd.join(within_means)
    return df