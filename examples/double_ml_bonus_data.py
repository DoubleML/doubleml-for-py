"""
===============
DML: Bonus Data
===============
This example shows
TODO: Add a general description!
"""
print(__doc__)

# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.linalg import toeplitz

from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import doubleml as dml
from doubleml.datasets import fetch_bonus


# %%
#

plt.rcParams['figure.figsize'] = 14, 6
sns.set()

# %%
# Load bonus data using the dml datasets module
# ---------------------------------------------

raw_data = dml.datasets.fetch_bonus()
raw_data.head()


# %%
# Data preprocessing
# ------------------

# data transformations and subselection
ind = (raw_data['tg'] == 0) | (raw_data['tg'] == 4)
data = raw_data.copy()[ind]
data.reset_index(inplace=True)
data['tg'].replace(4, 1, inplace=True)
data['inuidur1'] = np.log(data['inuidur1'])


# %%
#

# variable dep as factor (dummy encoding)
dummy_enc = OneHotEncoder(drop='first', categories='auto').fit(data.loc[:, ['dep']])
xx = dummy_enc.transform(data.loc[:, ['dep']]).toarray()
data['dep1'] = xx[:,0]
data['dep2'] = xx[:,1]

# %%
#

y_col = 'inuidur1'
d_cols = ['tg']
x_cols = ['female', 'black', 'othrace',
          'dep1', 'dep2',
          'q2', 'q3', 'q4', 'q5', 'q6',
          'agelt35', 'agegt54', 'durable', 'lusd', 'husd']
dml_data = dml.DoubleMLData(data, y_col, d_cols, x_cols)
dml_data


# %%
# Specify learner and estimate causal parameter
# ---------------------------------------------

# Set machine learning methods for m & g
learner = RandomForestRegressor(max_depth=2, n_estimators=100)
ml_g = clone(learner)
ml_m = clone(learner)
n_folds = 2
n_rep = 100

dml_plr_obj_rf = dml.DoubleMLPLR(dml_data,
                                 ml_g,
                                 ml_m,
                                 n_folds,
                                 n_rep,
                                 'IV-type',
                                 'dml1')

# %%
#

dml_plr_obj_rf.fit()
dml_plr_obj_rf.summary

# %%
#

poly = PolynomialFeatures(2, include_bias=False)
data_transf = poly.fit_transform(data[x_cols])
x_cols_lasso = poly.get_feature_names(x_cols)

data_transf = pd.DataFrame(data_transf, columns=x_cols_lasso)
data_transf = pd.concat((data[[y_col] + d_cols], data_transf),
                        axis=1, sort=False)

dml_data_lasso = dml.DoubleMLData(data_transf, y_col, d_cols, x_cols_lasso)
dml_data_lasso

# %%
#

# Set machine learning methods for m & g
learner = Lasso(alpha=0.1)
ml_g = clone(learner)
ml_m = clone(learner)
n_folds = 2
n_rep = 100

dml_plr_obj_lasso = dml.DoubleMLPLR(dml_data_lasso,
                                    ml_g,
                                    ml_m,
                                    n_folds,
                                    n_rep,
                                    'partialling out',
                                    'dml2')

# %%
#

dml_plr_obj_lasso.fit()
dml_plr_obj_lasso.summary

# %%
#

# Set machine learning methods for m & g
ml_g = RandomForestRegressor(max_depth=2, n_estimators=100)
ml_m = RandomForestClassifier(max_depth=2, n_estimators=100)
n_folds = 2
n_rep=100

dml_irm_obj = dml.DoubleMLIRM(dml_data,
                              ml_g,
                              ml_m,
                              n_folds,
                              n_rep,
                              'ATE',
                              'dml2')

# %%
#

dml_irm_obj.fit()
dml_irm_obj.summary
