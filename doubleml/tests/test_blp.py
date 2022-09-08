import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import doubleml as dml

from ._utils_blp_manual import fit_blp, blp_confint


@pytest.fixture(scope='module',
                params=[True, False])
def ci_joint(request):
    return request.param

@pytest.fixture(scope='module',
                params=[0.95, 0.9])
def ci_level(request):
    return request.param


@pytest.fixture(scope='module')
def dml_blp_fixture(ci_joint, ci_level):
    np.random.seed(42)
    n = 200

    # collect data
    np.random.seed(42)
    obj_dml_data = dml.datasets.make_irm_data(n_obs=n, dim_x=5)

    # First stage estimation
    ml_m = RandomForestRegressor(n_estimators=100)
    ml_g = RandomForestClassifier(n_estimators=100)

    dml_irm_obj = dml.DoubleMLIRM(obj_dml_data,
                                  ml_g=ml_m,
                                  ml_m=ml_g,
                                  trimming_threshold=0.05,
                                  n_folds=5)

    dml_irm_obj.fit()

    # create a random basis
    random_basis = pd.DataFrame(np.random.normal(0, 1, size=(n, 5)))

    # cate
    cate = dml_irm_obj.cate(random_basis)

    # get the orthogonal signal from the IRM model
    orth_signal = dml_irm_obj.psi_b.reshape(-1)
    cate_manual = fit_blp(orth_signal, random_basis)

    np.random.seed(42)
    ci = cate.confint(random_basis, joint=ci_joint, level=ci_level, n_rep_boot=1000)
    np.random.seed(42)
    ci_manual = blp_confint(cate_manual, random_basis, joint=ci_joint, level=ci_level, n_rep_boot=1000)

    res_dict = {'coef': cate.blp_model.params,
                'coef_manual': cate_manual.params,
                'values': cate.blp_model.fittedvalues,
                'values_manual':  cate_manual.fittedvalues,
                'omega': cate.blp_omega,
                'omega_manual': cate_manual.cov_HC0,
                'ci': ci,
                'ci_manual': ci_manual}

    return res_dict


@pytest.mark.ci
def test_dml_blp_coef(dml_blp_fixture):
    assert np.allclose(dml_blp_fixture['coef'],
                       dml_blp_fixture['coef_manual'],
                       rtol=1e-9, atol=1e-4)

@pytest.mark.ci
def test_dml_blp_values(dml_blp_fixture):
    assert np.allclose(dml_blp_fixture['values'],
                       dml_blp_fixture['values_manual'],
                       rtol=1e-9, atol=1e-4)

@pytest.mark.ci
def test_dml_blp_omega(dml_blp_fixture):
    assert np.allclose(dml_blp_fixture['omega'],
                       dml_blp_fixture['omega_manual'],
                       rtol=1e-9, atol=1e-4)

@pytest.mark.ci
def test_dml_blp_ci(dml_blp_fixture):
    assert np.allclose(dml_blp_fixture['ci'],
                       dml_blp_fixture['ci_manual'],
                       rtol=1e-9, atol=1e-4)

