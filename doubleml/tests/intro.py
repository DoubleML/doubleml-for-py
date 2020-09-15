from doubleml.datasets import fetch_401K
# Load data
df_401k = fetch_401K()
df_401k.head(5)

from doubleml import DoubleMLData
# Specify the data and the variables for the causal model
obj_dml_data_401k = DoubleMLData(df_401k,
                                 y_col='net_tfa',
                                 d_cols='e401',
                                 x_cols=['age', 'inc', 'educ', 'fsize', 'marr',
                                         'twoearn', 'db', 'pira', 'hown'])
print(obj_dml_data_401k)

import numpy as np
# Generate data
n_obs = 500
n_vars = 100
theta = 3
X = np.random.normal(size=(n_obs, n_vars))
d = np.dot(X[:, :3], np.array([5, 5, 5])) + np.random.standard_normal(size=(n_obs,))
y = theta * d + np.dot(X[:, :3], np.array([5, 5, 5])) + np.random.standard_normal(size=(n_obs,))

from doubleml import DoubleMLData
obj_dml_data_sim = DoubleMLData.from_arrays(X, y, d)
print(obj_dml_data_sim)

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
learner = RandomForestRegressor(max_depth=2, n_estimators=100)
ml_learners_401k = {'ml_m': clone(learner),
                    'ml_g': clone(learner)}
learner = Lasso(alpha=np.sqrt(np.log(n_vars)/(n_obs)))
ml_learners_sim = {'ml_m': clone(learner),
                   'ml_g': clone(learner)}

from doubleml import DoubleMLPLR
obj_dml_plr_401k = DoubleMLPLR(obj_dml_data_401k, ml_learners_401k)
obj_dml_plr_sim = DoubleMLPLR(obj_dml_data_sim, ml_learners_sim)

obj_dml_plr_401k.fit()
print(obj_dml_plr_401k.summary)
obj_dml_plr_sim.fit()
print(obj_dml_plr_sim.summary)


