import pytest
import numpy as np
import pandas as pd

from doubleml import DoubleMLData, DoubleMLPLR
from doubleml.datasets import make_plr_CCDDHNR2018, _make_pliv_data, make_pliv_CHS2015
from sklearn.linear_model import Lasso


@pytest.fixture(scope="module")
def dml_data_fixture(generate_data1):
    data = generate_data1
    np.random.seed(3141)
    x_cols = data.columns[data.columns.str.startswith('X')].tolist()

    obj_from_np = DoubleMLData.from_arrays(data.loc[:, x_cols].values,
                                           data['y'].values, data['d'].values)

    obj_from_pd = DoubleMLData(data, 'y', ['d'], x_cols)

    return {'obj_from_np': obj_from_np,
            'obj_from_pd': obj_from_pd}


@pytest.mark.ci
def test_dml_data_x(dml_data_fixture):
    assert np.allclose(dml_data_fixture['obj_from_np'].x,
                       dml_data_fixture['obj_from_pd'].x,
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_data_y(dml_data_fixture):
    assert np.allclose(dml_data_fixture['obj_from_np'].y,
                       dml_data_fixture['obj_from_pd'].y,
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_data_d(dml_data_fixture):
    assert np.allclose(dml_data_fixture['obj_from_np'].d,
                       dml_data_fixture['obj_from_pd'].d,
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_obj_vs_from_arrays():
    np.random.seed(3141)
    dml_data = make_plr_CCDDHNR2018(n_obs=100)
    dml_data_from_array = DoubleMLData.from_arrays(dml_data.data[dml_data.x_cols],
                                                   dml_data.data[dml_data.y_col],
                                                   dml_data.data[dml_data.d_cols])
    assert dml_data_from_array.data.equals(dml_data.data)

    dml_data = _make_pliv_data(n_obs=100)
    dml_data_from_array = DoubleMLData.from_arrays(dml_data.data[dml_data.x_cols],
                                                   dml_data.data[dml_data.y_col],
                                                   dml_data.data[dml_data.d_cols],
                                                   dml_data.data[dml_data.z_cols])
    assert dml_data_from_array.data.equals(dml_data.data)

    dml_data = make_pliv_CHS2015(n_obs=100, dim_z=5)
    dml_data_from_array = DoubleMLData.from_arrays(dml_data.data[dml_data.x_cols],
                                                   dml_data.data[dml_data.y_col],
                                                   dml_data.data[dml_data.d_cols],
                                                   dml_data.data[dml_data.z_cols])
    assert np.array_equal(dml_data_from_array.data, dml_data.data)  # z_cols name differ

    dml_data = make_plr_CCDDHNR2018(n_obs=100)
    df = dml_data.data.copy().iloc[:, :10]
    df.columns = [f'X{i+1}' for i in np.arange(7)] + ['y', 'd1', 'd2']
    dml_data = DoubleMLData(df, 'y', ['d1', 'd2'], [f'X{i+1}' for i in np.arange(7)])
    dml_data_from_array = DoubleMLData.from_arrays(dml_data.data[dml_data.x_cols],
                                                   dml_data.data[dml_data.y_col],
                                                   dml_data.data[dml_data.d_cols])
    assert np.array_equal(dml_data_from_array.data, dml_data.data)


@pytest.mark.ci
def test_add_vars_in_df():
    # additional variables in the df shouldn't affect results
    np.random.seed(3141)
    df = make_plr_CCDDHNR2018(n_obs=100, return_type='DataFrame')
    dml_data_full_df = DoubleMLData(df, 'y', 'd', ['X1', 'X11', 'X13'])
    dml_data_subset = DoubleMLData(df[['X1', 'X11', 'X13'] + ['y', 'd']], 'y', 'd', ['X1', 'X11', 'X13'])
    dml_plr_full_df = DoubleMLPLR(dml_data_full_df, Lasso(), Lasso())
    dml_plr_subset = DoubleMLPLR(dml_data_subset, Lasso(), Lasso(), draw_sample_splitting=False)
    dml_plr_subset.set_sample_splitting(dml_plr_full_df.smpls)
    dml_plr_full_df.fit()
    dml_plr_subset.fit()
    assert np.allclose(dml_plr_full_df.coef, dml_plr_subset.coef, rtol=1e-9, atol=1e-4)
    assert np.allclose(dml_plr_full_df.se, dml_plr_subset.se, rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_data_no_instr():
    np.random.seed(3141)
    dml_data = make_plr_CCDDHNR2018(n_obs=100)
    assert dml_data.z is None
    assert dml_data.n_instr == 0

    x, y, d = make_plr_CCDDHNR2018(n_obs=100, return_type='array')
    dml_data = DoubleMLData.from_arrays(x, y, d)
    assert dml_data.z is None
    assert dml_data.n_instr == 0


@pytest.mark.ci
def test_x_cols_setter_defaults():
    df = pd.DataFrame(np.tile(np.arange(4), (4, 1)),
                      columns=['yy', 'dd', 'xx1', 'xx2'])
    dml_data = DoubleMLData(df, y_col='yy', d_cols='dd')
    assert dml_data.x_cols == ['xx1', 'xx2']


@pytest.mark.ci
def test_x_cols_setter():
    np.random.seed(3141)
    dml_data = make_plr_CCDDHNR2018(n_obs=100)
    orig_x_cols = dml_data.x_cols

    # check that after changing the x_cols, the x array gets updated
    x_comp = dml_data.data[['X1', 'X11', 'X13']].values
    dml_data.x_cols = ['X1', 'X11', 'X13']
    assert np.array_equal(dml_data.x, x_comp)

    msg = 'Invalid covariates x_cols. At least one covariate is no data column.'
    with pytest.raises(ValueError, match=msg):
        dml_data.x_cols = ['X1', 'X11', 'A13']

    msg = (r'The covariates x_cols must be of str or list type \(or None\). '
           "5 of type <class 'int'> was passed.")
    with pytest.raises(TypeError, match=msg):
        dml_data.x_cols = 5

    # check single covariate
    x_comp = dml_data.data[['X13']].values
    dml_data.x_cols = 'X13'
    assert np.array_equal(dml_data.x, x_comp)

    # check setting None brings us back to orig_x_cols
    x_comp = dml_data.data[orig_x_cols].values
    dml_data.x_cols = None
    assert np.array_equal(dml_data.x, x_comp)


@pytest.mark.ci
def test_d_cols_setter():
    np.random.seed(3141)
    dml_data = make_plr_CCDDHNR2018(n_obs=100)
    df = dml_data.data.copy().iloc[:, :10]
    df.columns = [f'X{i + 1}' for i in np.arange(7)] + ['y', 'd1', 'd2']
    dml_data = DoubleMLData(df, 'y', ['d1', 'd2'], [f'X{i + 1}' for i in np.arange(7)])

    # check that after changing d_cols, the d array gets updated
    d_comp = dml_data.data['d2'].values
    dml_data.d_cols = ['d2', 'd1']
    assert np.array_equal(dml_data.d, d_comp)

    msg = r'Invalid treatment variable\(s\) d_cols. At least one treatment variable is no data column.'
    with pytest.raises(ValueError, match=msg):
        dml_data.d_cols = ['d1', 'd13']
    with pytest.raises(ValueError, match=msg):
        dml_data.d_cols = 'd13'

    msg = (r'The treatment variable\(s\) d_cols must be of str or list type. '
           "5 of type <class 'int'> was passed.")
    with pytest.raises(TypeError, match=msg):
        dml_data.d_cols = 5

    # check single covariate
    d_comp = dml_data.data['d2'].values
    dml_data.d_cols = 'd2'
    assert np.array_equal(dml_data.d, d_comp)
    assert dml_data.n_treat == 1


@pytest.mark.ci
def test_z_cols_setter():
    np.random.seed(3141)
    dml_data = make_plr_CCDDHNR2018(n_obs=100)
    df = dml_data.data.copy().iloc[:, :10]
    df.columns = [f'X{i + 1}' for i in np.arange(4)] + [f'z{i + 1}' for i in np.arange(3)] + ['y', 'd1', 'd2']
    dml_data = DoubleMLData(df, 'y', ['d1', 'd2'],
                            [f'X{i + 1}' for i in np.arange(4)],
                            [f'z{i + 1}' for i in np.arange(3)])

    # check that after changing z_cols, the z array gets updated
    z_comp = dml_data.data[['z1', 'z2']].values
    dml_data.z_cols = ['z1', 'z2']
    assert np.array_equal(dml_data.z, z_comp)

    msg = r'Invalid instrumental variable\(s\) z_cols. At least one instrumental variable is no data column.'
    with pytest.raises(ValueError, match=msg):
        dml_data.z_cols = ['z1', 'a13']
    with pytest.raises(ValueError, match=msg):
        dml_data.z_cols = 'a13'

    msg = (r'The instrumental variable\(s\) z_cols must be of str or list type \(or None\). '
           "5 of type <class 'int'> was passed.")
    with pytest.raises(TypeError, match=msg):
        dml_data.z_cols = 5

    # check single instrument
    z_comp = dml_data.data[['z2']].values
    dml_data.z_cols = 'z2'
    assert np.array_equal(dml_data.z, z_comp)

    # check None
    dml_data.z_cols = None
    assert dml_data.n_instr == 0
    assert dml_data.z is None


@pytest.mark.ci
def test_y_col_setter():
    np.random.seed(3141)
    dml_data = make_plr_CCDDHNR2018(n_obs=100)
    df = dml_data.data.copy().iloc[:, :10]
    df.columns = [f'X{i + 1}' for i in np.arange(7)] + ['y', 'y123', 'd']
    dml_data = DoubleMLData(df, 'y', 'd', [f'X{i + 1}' for i in np.arange(7)])

    # check that after changing y_col, the y array gets updated
    y_comp = dml_data.data['y123'].values
    dml_data.y_col = 'y123'
    assert np.array_equal(dml_data.y, y_comp)

    msg = r'Invalid outcome variable y_col. d13 is no data column.'
    with pytest.raises(ValueError, match=msg):
        dml_data.y_col = 'd13'

    msg = (r'The outcome variable y_col must be of str type. '
           "5 of type <class 'int'> was passed.")
    with pytest.raises(TypeError, match=msg):
        dml_data.y_col = 5


@pytest.mark.ci
def test_use_other_treat_as_covariate():
    np.random.seed(3141)
    dml_data = make_plr_CCDDHNR2018(n_obs=100)
    df = dml_data.data.copy().iloc[:, :10]
    df.columns = [f'X{i + 1}' for i in np.arange(7)] + ['y', 'd1', 'd2']
    dml_data = DoubleMLData(df, 'y', ['d1', 'd2'], [f'X{i + 1}' for i in np.arange(7)],
                            use_other_treat_as_covariate=True)
    dml_data.set_x_d('d1')
    assert np.array_equal(dml_data.d, df['d1'].values)
    assert np.array_equal(dml_data.x, df[[f'X{i + 1}' for i in np.arange(7)] + ['d2']].values)
    dml_data.set_x_d('d2')
    assert np.array_equal(dml_data.d, df['d2'].values)
    assert np.array_equal(dml_data.x, df[[f'X{i + 1}' for i in np.arange(7)] + ['d1']].values)

    dml_data = DoubleMLData(df, 'y', ['d1', 'd2'], [f'X{i + 1}' for i in np.arange(7)],
                            use_other_treat_as_covariate=False)
    dml_data.set_x_d('d1')
    assert np.array_equal(dml_data.d, df['d1'].values)
    assert np.array_equal(dml_data.x, df[[f'X{i + 1}' for i in np.arange(7)]].values)
    dml_data.set_x_d('d2')
    assert np.array_equal(dml_data.d, df['d2'].values)
    assert np.array_equal(dml_data.x, df[[f'X{i + 1}' for i in np.arange(7)]].values)

    msg = 'use_other_treat_as_covariate must be True or False. Got 1.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLData(df, 'y', ['d1', 'd2'], [f'X{i + 1}' for i in np.arange(7)],
                         use_other_treat_as_covariate=1)

    msg = 'Invalid treatment_var. d3 is not in d_cols.'
    with pytest.raises(ValueError, match=msg):
        dml_data.set_x_d('d3')

    msg = r"treatment_var must be of str type. \['d1', 'd2'\] of type <class 'list'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_data.set_x_d(['d1', 'd2'])


@pytest.mark.ci
def test_disjoint_sets():
    np.random.seed(3141)
    df = pd.DataFrame(np.tile(np.arange(4), (4, 1)),
                      columns=['yy', 'dd1', 'xx1', 'xx2'])

    msg = (r'At least one variable/column is set as treatment variable \(``d_cols``\) and as covariate\(``x_cols``\). '
           'Consider using parameter ``use_other_treat_as_covariate``.')
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(df, y_col='yy', d_cols=['dd1', 'xx1'], x_cols=['xx1', 'xx2'])
    msg = 'yy cannot be set as outcome variable ``y_col`` and treatment variable in ``d_cols``'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(df, y_col='yy', d_cols=['dd1', 'yy'], x_cols=['xx1', 'xx2'])
    msg = 'yy cannot be set as outcome variable ``y_col`` and covariate in ``x_cols``'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(df, y_col='yy', d_cols=['dd1'], x_cols=['xx1', 'yy', 'xx2'])
    msg = 'yy cannot be set as outcome variable ``y_col`` and instrumental variable in ``z_cols``'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(df, y_col='yy', d_cols=['dd1'], x_cols=['xx1', 'xx2'], z_cols='yy')
    msg = (r'At least one variable/column is set as treatment variable \(``d_cols``\) and instrumental variable in '
           '``z_cols``.')
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(df, y_col='yy', d_cols=['dd1'], x_cols=['xx1', 'xx2'], z_cols=['dd1'])
    msg = (r'At least one variable/column is set as covariate \(``x_cols``\) and instrumental variable in '
           '``z_cols``.')
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(df, y_col='yy', d_cols=['dd1'], x_cols=['xx1', 'xx2'], z_cols='xx2')


@pytest.mark.ci
def test_duplicates():
    np.random.seed(3141)
    dml_data = make_plr_CCDDHNR2018(n_obs=100)

    msg = r'Invalid treatment variable\(s\) d_cols: Contains duplicate values.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(dml_data.data, y_col='y', d_cols=['d', 'd', 'X1'], x_cols=['X3', 'X2'])
    with pytest.raises(ValueError, match=msg):
        dml_data.d_cols = ['d', 'd', 'X1']

    msg = 'Invalid covariates x_cols: Contains duplicate values.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(dml_data.data, y_col='y', d_cols=['d'], x_cols=['X3', 'X2', 'X3'])
    with pytest.raises(ValueError, match=msg):
        dml_data.x_cols=['X3', 'X2', 'X3']

    msg = r'Invalid instrumental variable\(s\) z_cols: Contains duplicate values.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(dml_data.data, y_col='y', d_cols=['d'], x_cols=['X3', 'X2'],
                         z_cols=['X15', 'X12', 'X12', 'X15'])
    with pytest.raises(ValueError, match=msg):
        dml_data.z_cols = ['X15', 'X12', 'X12', 'X15']

    msg = 'Invalid pd.DataFrame: Contains duplicate column names.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLData(pd.DataFrame(np.zeros((100, 5)), columns=['y', 'd', 'X3', 'X2', 'y']),
                         y_col='y', d_cols=['d'], x_cols=['X3', 'X2'])


@pytest.mark.ci
def test_dml_datatype():
    data_array = np.zeros((100, 10))
    # msg = ('data must be of pd.DataFrame type. '
    #        f'{str(data_array)} of type {str(type(data_array))} was passed.')
    with pytest.raises(TypeError):
        _ = DoubleMLData(data_array, y_col='y', d_cols=['d'], x_cols=['X3', 'X2'])
