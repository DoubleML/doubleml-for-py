import numpy as np
import pytest

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import doubleml as dml

from ._utils_blp_manual import fit_blp, blp_confint, create_spline_basis, create_synthetic_data


@pytest.fixture(scope='module',
                params=[True])
def constant(request):
    return request.param


@pytest.fixture(scope='module',
                params=[True, False])
def ci_joint(request):
    return request.param


@pytest.fixture(scope='module')
def dml_blp_fixture(constant, ci_joint):
    np.random.seed(42)
    n = 200
    n_w = 10
    support_size = 5
    n_x = 1

    # Create data
    data, covariates = create_synthetic_data(n=n, n_w=n_w, support_size=support_size, n_x=n_x, constant=constant)
    data_dml_base = dml.DoubleMLData(data,
                                     y_col='y',
                                     d_cols='t',
                                     x_cols=covariates)

    # First stage estimation
    ml_m = RandomForestRegressor(n_estimators=100)
    ml_g = RandomForestClassifier(n_estimators=100)

    np.random.seed(42)
    dml_irm = dml.DoubleMLIRM(data_dml_base,
                              ml_g=ml_m,
                              ml_m=ml_g,
                              trimming_threshold=0.05,
                              n_folds=5)

    dml_irm.fit(store_predictions=True)

    spline_basis = create_spline_basis(X=data["x"])

    cate = dml.double_ml_blp.DoubleMLIRMBLP(dml_irm, basis=spline_basis).fit()
    cate_manual = fit_blp(dml_irm, spline_basis)

    np.random.seed(42)
    ci = cate.confint(spline_basis, joint=ci_joint, level=0.95, n_rep_boot=1000)
    np.random.seed(42)
    ci_manual = blp_confint(dml_irm, cate_manual, spline_basis, joint=ci_joint, level=0.95, n_rep_boot=1000)

    res_dict = {'coef': cate.blp_model.params,
                'coef_manual': cate_manual.params,
                'values': cate.blp_model.fittedvalues,
                'values_manual':  cate_manual.fittedvalues,
                'omega': cate.blp_omega,
                'omega_manual': cate_manual.cov_HC0,
                'ci': ci,
                'ci_manual': ci_manual}

    return res_dict


def test_dml_blp_coef(dml_blp_fixture):
    assert np.allclose(dml_blp_fixture['coef'],
                       dml_blp_fixture['coef_manual'],
                       rtol=1e-9, atol=1e-4)


def test_dml_blp_values(dml_blp_fixture):
    assert np.allclose(dml_blp_fixture['values'],
                       dml_blp_fixture['values_manual'],
                       rtol=1e-9, atol=1e-4)


def test_dml_blp_omega(dml_blp_fixture):
    assert np.allclose(dml_blp_fixture['omega'],
                       dml_blp_fixture['omega_manual'],
                       rtol=1e-9, atol=1e-4)

def test_dml_blp_ci(dml_blp_fixture):
    assert np.allclose(dml_blp_fixture['ci'],
                       dml_blp_fixture['ci_manual'],
                       rtol=1e-9, atol=1e-4)
