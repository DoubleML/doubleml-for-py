import numpy as np
import pytest
from sklearn.linear_model import Lasso

import doubleml as dml
from doubleml.plm.datasets import make_plr_CCDDHNR2018


@pytest.mark.ci
@pytest.mark.parametrize("n_rep, n_treat", [(1, 1), (2, 1), (1, 2), (3, 2)])
def test_tune_on_folds_result_per_treatment_and_repetition(n_rep, n_treat):
    np.random.seed(3141)
    df = make_plr_CCDDHNR2018(n_obs=200, dim_x=5, return_type="DataFrame")
    d_cols = ["d", "X1"][:n_treat]
    x_cols = [c for c in df.columns if c.startswith("X") and c not in d_cols]
    dml_data = dml.DoubleMLData(df, "y", d_cols, x_cols)
    dml_plr = dml.DoubleMLPLR(dml_data, Lasso(), Lasso(), n_folds=2, n_rep=n_rep)

    grid = {"ml_l": {"alpha": [0.05, 0.1]}, "ml_m": {"alpha": [0.05, 0.1]}}
    tune_res = dml_plr.tune(grid, tune_on_folds=True, n_folds_tune=2, return_tune_res=True)

    assert len(tune_res) == n_treat
    for i_d in range(n_treat):
        assert len(tune_res[i_d]) == n_rep
        assert len({id(res) for res in tune_res[i_d]}) == n_rep
        for i_rep in range(n_rep):
            # the results of each treatment and repetition are the parameters that were set
            params = dml_plr.params["ml_l"][d_cols[i_d]][i_rep]
            assert tune_res[i_d][i_rep]["params"]["ml_l"] == params
    dml_plr.fit()
