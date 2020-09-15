"""
===========================
Multiway Cluster Robust DML
===========================

This example shows how the multiway cluster roboust DML (Chiang et al. 2020) can be implemented with the DoubleML
package.
Chiang et al. (2020) consider double-indexed data

.. math::

    \\lbrace W_{ij}: i \\in \\lbrace 1, \\ldots, N \\rbrace, j \\in \\lbrace 1, \\ldots, M \\rbrace \\rbrace

and the partially linear IV regression model (PLIV)

.. math::
    Y_{ij} = D_{ij} \\theta_0 +  g_0(X_{ij}) + \\epsilon_{ij}, & &\\mathbb{E}(\\epsilon_{ij} | X_{ij}, Z_{ij}) = 0,

    Z_{ij} = m_0(X_{ij}) + v_{ij}, & &\\mathbb{E}(v_{ij} | X_{ij}) = 0.

TODO: Add a few more details and the reference!
https://arxiv.org/pdf/1909.03489.pdf
"""
print(__doc__)

# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.base import clone

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from doubleml import DoubleMLData, DoubleMLPLIV
from doubleml.double_ml_resampling import DoubleMLMultiwayResampling

from doubleml.datasets import make_pliv_multiway_cluster_data

# %%
# Simulate multiway cluster data
# ------------------------------
#
# We use the PLIV data generating process described in Section 4.1 of Chiang et al. (2020).

# %%
#

# Set the simulation parameters
N = 25  # number of observations (first dimension)
M = 25  # number of observations (second dimension)
dim_X = 100  # dimension of X

data = make_pliv_multiway_cluster_data(N, M, dim_X)

# %%
#

# The data comes with multi index for rows (tuples with two entries)
data.head(30)


# %%
# Initialize the objects of class DoubleMLData and DoubleMLPLIV
# -------------------------------------------------------------

# collect data and specify the DoubleMLData object
x_cols = data.columns[data.columns.str.startswith('x')].tolist()
obj_dml_data = DoubleMLData(data, 'Y', 'D', x_cols, 'Z')

# Set machine learning methods for m & g
learner = RandomForestRegressor(max_depth=2, n_estimators=10)
ml_learners = {'ml_m': clone(learner),
               'ml_g': clone(learner),
               'ml_r': clone(learner)}

# initialize the DoubleMLPLIV object
dml_pliv_obj = DoubleMLPLIV(obj_dml_data,
                            ml_learners,
                            inf_model='partialling out',
                            dml_procedure='dml1',
                            draw_sample_splitting=False)


# %%
# Split samples and transfer the sample splitting to the object
# -------------------------------------------------------------

K = 3  # number of folds
smpl_sizes = [N, M]
obj_dml_multiway_resampling = DoubleMLMultiwayResampling(K, smpl_sizes)
smpls_multi_ind, smpls_lin_ind = obj_dml_multiway_resampling.split_samples()

dml_pliv_obj.set_sample_splitting([smpls_lin_ind])


# %%
# Fit the model and show a summary
# --------------------------------

dml_pliv_obj.fit()
print(dml_pliv_obj.summary)

# %%
# Visualization of sample splitting with tuple and linear indexing
# ----------------------------------------------------------------

#discrete color scheme
x = sns.color_palette("RdBu_r", 7)
cMap = ListedColormap([x[0], x[3], x[6]])
plt.rcParams['figure.figsize'] = 15, 12
sns.set(font_scale=1.3)

# %%
# Visualize sample splitting with tuples (one plot per fold)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

for i_split, this_split_ind in enumerate(smpls_multi_ind):
    plt.subplot(K, K, i_split + 1)
    df = pd.DataFrame(np.zeros([N, M]))
    ind_array_train = np.array([*this_split_ind[0]])
    ind_array_test = np.array([*this_split_ind[1]])
    df.loc[ind_array_train[:, 0], ind_array_train[:, 1]] = -1.
    df.loc[ind_array_test[:, 0], ind_array_test[:, 1]] = 1.

    ax = sns.heatmap(df, cmap=cMap);
    ax.invert_yaxis();
    ax.set_ylim([0, M]);
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([-0.667, 0, 0.667])
    if i_split % K == (K - 1):
        colorbar.set_ticklabels(['Nuisance', '', 'Score'])
    else:
        colorbar.set_ticklabels(['', '', ''])


# %%
# Visualize sample splitting with linear indexing (one column per fold)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

df = pd.DataFrame(np.zeros([N*M, K*K]))
for i_split, this_split_ind in enumerate(smpls_lin_ind):
    df.loc[this_split_ind[0], i_split] = -1.
    df.loc[this_split_ind[1], i_split] = 1.

ax = sns.heatmap(df, cmap=cMap);
ax.invert_yaxis();
ax.set_ylim([0, N*M]);
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([-0.667, 0, 0.667])
colorbar.set_ticklabels(['Nuisance', '', 'Score'])

