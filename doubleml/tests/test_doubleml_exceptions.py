import pytest
import pandas as pd
import numpy as np

from doubleml import DoubleMLPLR, DoubleMLIRM, DoubleMLIIVM, DoubleMLPLIV, DoubleMLData,\
    DoubleMLClusterData, DoubleMLPQ, DoubleMLLPQ, DoubleMLCVAR, DoubleMLQTE, DoubleMLDID, DoubleMLDIDCS
from doubleml.datasets import make_plr_CCDDHNR2018, make_irm_data, make_pliv_CHS2015, make_iivm_data, \
    make_pliv_multiway_cluster_CKMS2021, make_did_SZ2020
from doubleml.double_ml_data import DoubleMLBaseData

from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator

np.random.seed(3141)
n = 100
dml_data = make_plr_CCDDHNR2018(n_obs=n)
ml_l = Lasso()
ml_m = Lasso()
ml_g = Lasso()
ml_r = Lasso()
dml_plr = DoubleMLPLR(dml_data, ml_l, ml_m)
dml_plr_iv_type = DoubleMLPLR(dml_data, ml_l, ml_m, ml_g, score='IV-type')

dml_data_pliv = make_pliv_CHS2015(n_obs=n, dim_z=1)
dml_pliv = DoubleMLPLIV(dml_data_pliv, ml_l, ml_m, ml_r)

dml_data_irm = make_irm_data(n_obs=n)
dml_data_iivm = make_iivm_data(n_obs=n)
dml_cluster_data_pliv = make_pliv_multiway_cluster_CKMS2021(N=10, M=10)
dml_data_did = make_did_SZ2020(n_obs=n)
dml_data_did_cs = make_did_SZ2020(n_obs=n, cross_sectional_data=True)
(x, y, d, z) = make_iivm_data(n_obs=n, return_type="array")
y[y > 0] = 1
y[y < 0] = 0
dml_data_irm_binary_outcome = DoubleMLData.from_arrays(x, y, d)
dml_data_iivm_binary_outcome = DoubleMLData.from_arrays(x, y, d, z)


class DummyDataClass(DoubleMLBaseData):
    def __init__(self,
                 data):
        DoubleMLBaseData.__init__(self, data)

    @property
    def n_coefs(self):
        return 1


@pytest.mark.ci
def test_doubleml_exception_data():
    msg = 'The data must be of DoubleMLData or DoubleMLClusterData type.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPLR(pd.DataFrame(), ml_l, ml_m)

    msg = 'The data must be of DoubleMLData type.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPLR(DummyDataClass(pd.DataFrame(np.zeros((100, 10)))), ml_l, ml_m)
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPLIV(DummyDataClass(pd.DataFrame(np.zeros((100, 10)))), ml_l, ml_m, ml_r)
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLIRM(DummyDataClass(pd.DataFrame(np.zeros((100, 10)))), ml_g, ml_m)
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLIIVM(DummyDataClass(pd.DataFrame(np.zeros((100, 10)))), ml_g, ml_m, ml_r)
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPQ(DummyDataClass(pd.DataFrame(np.zeros((100, 10)))), ml_g, ml_m, treatment=1)
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLLPQ(DummyDataClass(pd.DataFrame(np.zeros((100, 10)))), ml_g, ml_m, treatment=1)
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLCVAR(DummyDataClass(pd.DataFrame(np.zeros((100, 10)))), ml_g, ml_m, treatment=1)
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLQTE(DummyDataClass(pd.DataFrame(np.zeros((100, 10)))), ml_g, ml_m)
    msg = 'For repeated outcomes the data must be of DoubleMLData type.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLDID(DummyDataClass(pd.DataFrame(np.zeros((100, 10)))), ml_g, ml_m)
    msg = 'For repeated cross sections the data must be of DoubleMLData type. '
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLDIDCS(DummyDataClass(pd.DataFrame(np.zeros((100, 10)))), ml_g, ml_m)

    # PLR with IV
    msg = (r'Incompatible data. Z1 have been set as instrumental variable\(s\). '
           'To fit a partially linear IV regression model use DoubleMLPLIV instead of DoubleMLPLR.')
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPLR(dml_data_pliv, ml_l, ml_m)

    # PLIV without IV
    msg = ('Incompatible data. '
           'At least one variable must be set as instrumental variable. '
           r'To fit a partially linear regression model without instrumental variable\(s\) '
           'use DoubleMLPLR instead of DoubleMLPLIV.')
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPLIV(dml_data, Lasso(), Lasso(), Lasso())

    # IRM with IV
    msg = (r'Incompatible data. z have been set as instrumental variable\(s\). '
           'To fit an interactive IV regression model use DoubleMLIIVM instead of DoubleMLIRM.')
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLIRM(dml_data_iivm, Lasso(), LogisticRegression())
    msg = ('Incompatible data. To fit an IRM model with DML exactly one binary variable with values 0 and 1 '
           'needs to be specified as treatment variable.')
    df_irm = dml_data_irm.data.copy()
    df_irm['d'] = df_irm['d'] * 2
    with pytest.raises(ValueError, match=msg):
        # non-binary D for IRM
        _ = DoubleMLIRM(DoubleMLData(df_irm, 'y', 'd'),
                        Lasso(), LogisticRegression())
    with pytest.raises(ValueError, match=msg):
        # multiple D for IRM
        _ = DoubleMLIRM(DoubleMLData(df_irm, 'y', ['d', 'X1']),
                        Lasso(), LogisticRegression())

    msg = ('Incompatible data. To fit an IIVM model with DML exactly one binary variable with values 0 and 1 '
           'needs to be specified as treatment variable.')
    df_iivm = dml_data_iivm.data.copy()
    df_iivm['d'] = df_iivm['d'] * 2
    with pytest.raises(ValueError, match=msg):
        # non-binary D for IIVM
        _ = DoubleMLIIVM(DoubleMLData(df_iivm, 'y', 'd', z_cols='z'),
                         Lasso(), LogisticRegression(), LogisticRegression())
    df_iivm = dml_data_iivm.data.copy()
    with pytest.raises(ValueError, match=msg):
        # multiple D for IIVM
        _ = DoubleMLIIVM(DoubleMLData(df_iivm, 'y', ['d', 'X1'], z_cols='z'),
                         Lasso(), LogisticRegression(), LogisticRegression())

    msg = ('Incompatible data. To fit an IIVM model with DML exactly one binary variable with values 0 and 1 '
           'needs to be specified as instrumental variable.')
    with pytest.raises(ValueError, match=msg):
        # IIVM without IV
        _ = DoubleMLIIVM(dml_data_irm,
                         Lasso(), LogisticRegression(), LogisticRegression())
    df_iivm = dml_data_iivm.data.copy()
    df_iivm['z'] = df_iivm['z'] * 2
    with pytest.raises(ValueError, match=msg):
        # non-binary Z for IIVM
        _ = DoubleMLIIVM(DoubleMLData(df_iivm, 'y', 'd', z_cols='z'),
                         Lasso(), LogisticRegression(), LogisticRegression())
    df_iivm = dml_data_iivm.data.copy()
    with pytest.raises(ValueError, match=msg):
        # multiple Z for IIVM
        _ = DoubleMLIIVM(DoubleMLData(df_iivm, 'y', 'd', z_cols=['z', 'X1']),
                         Lasso(), LogisticRegression(), LogisticRegression())

    # PQ with IV
    msg = r'Incompatible data. z have been set as instrumental variable\(s\).'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPQ(dml_data_iivm, LogisticRegression(), LogisticRegression(), treatment=1)
    msg = ('Incompatible data. To fit an PQ model with DML exactly one binary variable with values 0 and 1 '
           'needs to be specified as treatment variable.')
    df_irm = dml_data_irm.data.copy()
    df_irm['d'] = df_irm['d'] * 2
    with pytest.raises(ValueError, match=msg):
        # non-binary D for PQ
        _ = DoubleMLPQ(DoubleMLData(df_irm, 'y', 'd'),
                       LogisticRegression(), LogisticRegression(), treatment=1)
    df_irm = dml_data_irm.data.copy()
    with pytest.raises(ValueError, match=msg):
        # multiple D for PQ
        _ = DoubleMLPQ(DoubleMLData(df_irm, 'y', ['d', 'X1']),
                       LogisticRegression(), LogisticRegression(), treatment=1)

    # LPQ with non-binary treatment
    msg = ('Incompatible data. To fit an LPQ model with DML exactly one binary variable with values 0 and 1 '
           'needs to be specified as treatment variable.')
    df_iivm = dml_data_iivm.data.copy()
    df_iivm['d'] = df_iivm['d'] * 2
    with pytest.raises(ValueError, match=msg):
        # non-binary D for LPQ
        _ = DoubleMLLPQ(DoubleMLData(df_iivm, 'y', 'd', 'z'),
                        LogisticRegression(), LogisticRegression(), treatment=1)
    df_iivm = dml_data_iivm.data.copy()
    with pytest.raises(ValueError, match=msg):
        # multiple D for LPQ
        _ = DoubleMLLPQ(DoubleMLData(df_iivm, 'y', ['d', 'X1'], 'z'),
                        LogisticRegression(), LogisticRegression(), treatment=1)
    msg = ('Incompatible data. To fit an LPQ model with DML exactly one binary variable with values 0 and 1 '
           'needs to be specified as instrumental variable.')
    df_iivm = dml_data_iivm.data.copy()
    df_iivm['z'] = df_iivm['z'] * 2
    with pytest.raises(ValueError, match=msg):
        # non-binary Z for LPQ
        _ = DoubleMLLPQ(DoubleMLData(df_iivm, 'y', 'd', 'z'),
                        LogisticRegression(), LogisticRegression(), treatment=1)

    # CVAR with IV
    msg = r'Incompatible data. z have been set as instrumental variable\(s\).'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLCVAR(dml_data_iivm, Lasso(), LogisticRegression(), treatment=1)
    msg = ('Incompatible data. To fit an CVaR model with DML exactly one binary variable with values 0 and 1 '
           'needs to be specified as treatment variable.')
    df_irm = dml_data_irm.data.copy()
    df_irm['d'] = df_irm['d'] * 2
    with pytest.raises(ValueError, match=msg):
        # non-binary D for CVAR
        _ = DoubleMLCVAR(DoubleMLData(df_irm, 'y', 'd'),
                         Lasso(), LogisticRegression(), treatment=1)
    df_irm = dml_data_irm.data.copy()
    with pytest.raises(ValueError, match=msg):
        # multiple D for CVAR
        _ = DoubleMLCVAR(DoubleMLData(df_irm, 'y', ['d', 'X1']),
                         Lasso(), LogisticRegression(), treatment=1)

    # QTE
    msg = ('Incompatible data. To fit an PQ model with DML exactly one binary variable with values 0 and 1 '
           'needs to be specified as treatment variable.')
    df_irm = dml_data_irm.data.copy()
    df_irm['d'] = df_irm['d'] * 2
    with pytest.raises(ValueError, match=msg):
        # non-binary D for QTE
        _ = DoubleMLQTE(DoubleMLData(df_irm, 'y', 'd'),
                        LogisticRegression(), LogisticRegression())
    df_irm = dml_data_irm.data.copy()
    with pytest.raises(ValueError, match=msg):
        # multiple D for QTE
        _ = DoubleMLQTE(DoubleMLData(df_irm, 'y', ['d', 'X1']),
                        LogisticRegression(), LogisticRegression())

    # DID with IV
    msg = r'Incompatible data. z have been set as instrumental variable\(s\).'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLDID(dml_data_iivm, Lasso(), LogisticRegression())
    msg = ('Incompatible data. To fit an DID model with DML exactly one binary variable with values 0 and 1 '
           'needs to be specified as treatment variable.')
    df_irm = dml_data_irm.data.copy()
    df_irm['d'] = df_irm['d'] * 2
    with pytest.raises(ValueError, match=msg):
        # non-binary D for DID
        _ = DoubleMLDID(DoubleMLData(df_irm, 'y', 'd'),
                        Lasso(), LogisticRegression())
    df_irm = dml_data_irm.data.copy()
    with pytest.raises(ValueError, match=msg):
        # multiple D for DID
        _ = DoubleMLDID(DoubleMLData(df_irm, 'y', ['d', 'X1']),
                        Lasso(), LogisticRegression())

    # DIDCS with IV
    msg = r'Incompatible data. z have been set as instrumental variable\(s\).'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLDIDCS(dml_data_iivm, Lasso(), LogisticRegression())

    # DIDCS treatment exceptions
    msg = ('Incompatible data. To fit an DIDCS model with DML exactly one binary variable with values 0 and 1 '
           'needs to be specified as treatment variable.')
    df_did_cs = dml_data_did_cs.data.copy()
    df_did_cs['d'] = df_did_cs['d'] * 2
    with pytest.raises(ValueError, match=msg):
        # non-binary D for DIDCS
        _ = DoubleMLDIDCS(DoubleMLData(df_did_cs, y_col='y', d_cols='d', t_col='t'),
                          Lasso(), LogisticRegression())
    df_did_cs = dml_data_did_cs.data.copy()
    with pytest.raises(ValueError, match=msg):
        # multiple D for DIDCS
        _ = DoubleMLDIDCS(DoubleMLData(df_did_cs, y_col='y', d_cols=['d', 'Z1'], t_col='t'),
                          Lasso(), LogisticRegression())

    # DIDCS time exceptions
    msg = ('Incompatible data. To fit an DIDCS model with DML exactly one binary variable with values 0 and 1 '
           'needs to be specified as time variable.')
    df_did_cs = dml_data_did_cs.data.copy()
    df_did_cs['t'] = df_did_cs['t'] * 2
    with pytest.raises(ValueError, match=msg):
        # non-binary t for DIDCS
        _ = DoubleMLDIDCS(DoubleMLData(df_did_cs, y_col='y', d_cols='d', t_col='t'),
                          Lasso(), LogisticRegression())


@pytest.mark.ci
def test_doubleml_exception_scores():
    # PLR
    msg = 'Invalid score IV. Valid score IV-type or partialling out.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPLR(dml_data, ml_l, ml_m, score='IV')
    msg = 'score should be either a string or a callable. 0 was passed.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPLR(dml_data, ml_l, ml_m, score=0)

    # IRM
    msg = 'Invalid score IV. Valid score ATE or ATTE.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLIRM(dml_data_irm, Lasso(), LogisticRegression(), score='IV')
    msg = 'score should be either a string or a callable. 0 was passed.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLIRM(dml_data_irm, Lasso(), LogisticRegression(), score=0)

    # IIVM
    msg = 'Invalid score ATE. Valid score LATE.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLIIVM(dml_data_iivm, Lasso(), LogisticRegression(), LogisticRegression(), score='ATE')
    msg = 'score should be either a string or a callable. 0 was passed.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLIIVM(dml_data_iivm, Lasso(), LogisticRegression(), LogisticRegression(), score=0)

    # PLIV
    msg = 'Invalid score IV. Valid score partialling out.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPLIV(dml_data_pliv, Lasso(), Lasso(), Lasso(), score='IV')
    msg = 'score should be either a string or a callable. 0 was passed.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPLIV(dml_data_pliv, Lasso(), Lasso(), Lasso(), score=0)

    # PQ
    msg = 'Invalid score IV. Valid score PQ.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPQ(dml_data_irm, LogisticRegression(), LogisticRegression(), treatment=1, score='IV')
    msg = 'score should be a string. 2 was passed.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPQ(dml_data_irm, LogisticRegression(), LogisticRegression(), treatment=1, score=2)

    # LPQ
    msg = 'Invalid score IV. Valid score LPQ.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLLPQ(dml_data_iivm, LogisticRegression(), LogisticRegression(), treatment=1, score='IV')
    msg = 'score should be a string. 2 was passed.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLLPQ(dml_data_iivm, LogisticRegression(), LogisticRegression(), treatment=1, score=2)

    # CVaR
    msg = 'Invalid score IV. Valid score CVaR.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLCVAR(dml_data_irm, LogisticRegression(), LogisticRegression(), treatment=1, score='IV')
    msg = 'score should be a string. 2 was passed.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLCVAR(dml_data_irm, LogisticRegression(), LogisticRegression(), treatment=1, score=2)

    # QTE
    msg = 'Invalid score IV. Valid score PQ or LPQ or CVaR.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLQTE(dml_data_irm, LogisticRegression(), LogisticRegression(), score='IV')
    msg = 'score should be a string. 2 was passed.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLQTE(dml_data_irm, LogisticRegression(), LogisticRegression(), score=2)

    # DID
    msg = 'Invalid score IV. Valid score observational or experimental.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLDID(dml_data_did, Lasso(), LogisticRegression(), score='IV')
    msg = 'score should be a string. 2 was passed.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLDID(dml_data_did, Lasso(), LogisticRegression(), score=2)

    # DIDCS
    msg = 'Invalid score IV. Valid score observational or experimental.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLDIDCS(dml_data_did_cs, Lasso(), LogisticRegression(), score='IV')
    msg = 'score should be a string. 2 was passed.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLDIDCS(dml_data_did_cs, Lasso(), LogisticRegression(), score=2)


@pytest.mark.ci
def test_doubleml_exception_trimming_rule():
    msg = 'Invalid trimming_rule discard. Valid trimming_rule truncate.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLIRM(dml_data_irm, Lasso(), LogisticRegression(), trimming_rule='discard')
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLIIVM(dml_data_iivm, Lasso(), LogisticRegression(), LogisticRegression(), trimming_rule='discard')
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPQ(dml_data_irm, LogisticRegression(), LogisticRegression(), treatment=1, trimming_rule='discard')
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLLPQ(dml_data_iivm, LogisticRegression(), LogisticRegression(), treatment=1, trimming_rule='discard')
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLCVAR(dml_data_irm, LogisticRegression(), LogisticRegression(), treatment=1, trimming_rule='discard')
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLQTE(dml_data_irm, LogisticRegression(), LogisticRegression(), trimming_rule='discard')
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLDID(dml_data_did, Lasso(), LogisticRegression(), trimming_rule='discard')
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLDIDCS(dml_data_did_cs, Lasso(), LogisticRegression(), trimming_rule='discard')

    # check the trimming_threshold exceptions
    msg = "trimming_threshold has to be a float. Object of type <class 'str'> passed."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLIRM(dml_data_irm, Lasso(), LogisticRegression(),
                        trimming_rule='truncate', trimming_threshold="0.1")
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLIIVM(dml_data_iivm, Lasso(), LogisticRegression(), LogisticRegression(),
                         trimming_rule='truncate', trimming_threshold="0.1")
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPQ(dml_data_irm, LogisticRegression(), LogisticRegression(), treatment=1,
                       trimming_rule='truncate', trimming_threshold="0.1")
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLLPQ(dml_data_iivm, LogisticRegression(), LogisticRegression(), treatment=1,
                        trimming_rule='truncate', trimming_threshold="0.1")
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLCVAR(dml_data_irm, Lasso(), LogisticRegression(), treatment=1,
                         trimming_rule='truncate', trimming_threshold="0.1")
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLQTE(dml_data_irm, LogisticRegression(), LogisticRegression(),
                        trimming_rule='truncate', trimming_threshold="0.1")
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLDID(dml_data_did, Lasso(), LogisticRegression(),
                        trimming_rule='truncate', trimming_threshold="0.1")
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLDIDCS(dml_data_did_cs, Lasso(), LogisticRegression(),
                          trimming_rule='truncate', trimming_threshold="0.1")

    msg = 'Invalid trimming_threshold 0.6. trimming_threshold has to be between 0 and 0.5.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLIRM(dml_data_irm, Lasso(), LogisticRegression(),
                        trimming_rule='truncate', trimming_threshold=0.6)
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLIIVM(dml_data_iivm, Lasso(), LogisticRegression(), LogisticRegression(),
                         trimming_rule='truncate', trimming_threshold=0.6)
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPQ(dml_data_irm, LogisticRegression(), LogisticRegression(), treatment=1,
                       trimming_rule='truncate', trimming_threshold=0.6)
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLLPQ(dml_data_iivm, LogisticRegression(), LogisticRegression(), treatment=1,
                        trimming_rule='truncate', trimming_threshold=0.6)
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLCVAR(dml_data_irm, Lasso(), LogisticRegression(), treatment=1,
                         trimming_rule='truncate', trimming_threshold=0.6)
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLQTE(dml_data_irm, LogisticRegression(), LogisticRegression(),
                        trimming_rule='truncate', trimming_threshold=0.6)
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLDID(dml_data_did, Lasso(), LogisticRegression(),
                        trimming_rule='truncate', trimming_threshold=0.6)
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLDIDCS(dml_data_did_cs, Lasso(), LogisticRegression(),
                          trimming_rule='truncate', trimming_threshold=0.6)


@pytest.mark.ci
def test_doubleml_exception_quantiles():
    msg = "Quantile has to be a float. Object of type <class 'str'> passed."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPQ(dml_data_irm, ml_g, ml_m, treatment=1, quantile="0.4")
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLLPQ(dml_data_iivm, ml_g, ml_m, treatment=1, quantile="0.4")
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLCVAR(dml_data_irm, ml_g, ml_m, treatment=1, quantile="0.4")

    msg = "Quantile has be between 0 or 1. Quantile 1.0 passed."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPQ(dml_data_irm, ml_g, ml_m, treatment=1, quantile=1.)
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLLPQ(dml_data_iivm, ml_g, ml_m, treatment=1, quantile=1.)
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLCVAR(dml_data_irm, ml_g, ml_m, treatment=1, quantile=1.)

    msg = r'Quantiles have be between 0 or 1. Quantiles \[0.2 2. \] passed.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLQTE(dml_data_irm, ml_g, ml_m, quantiles=[0.2, 2])


@pytest.mark.ci
def test_doubleml_exception_treatment():
    msg = "Treatment indicator has to be an integer. Object of type <class 'str'> passed."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPQ(dml_data_irm, ml_g, ml_m, treatment="1")
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLLPQ(dml_data_iivm, ml_g, ml_m, treatment="1")
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLCVAR(dml_data_irm, ml_g, ml_m, treatment="1")

    msg = "Treatment indicator has be either 0 or 1. Treatment indicator 2 passed."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPQ(dml_data_irm, ml_g, ml_m, treatment=2)
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLLPQ(dml_data_iivm, ml_g, ml_m, treatment=2)
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLCVAR(dml_data_irm, ml_g, ml_m, treatment=2)


@pytest.mark.ci
def test_doubleml_exception_kde():
    msg = "kde should be either a callable or None. '0.1' was passed."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPQ(dml_data_irm, ml_g, ml_m, treatment=1, kde="0.1")
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLLPQ(dml_data_iivm, ml_g, ml_m, treatment=1, kde="0.1")


@pytest.mark.ci
def test_doubleml_exception_normalization():
    msg = "Normalization indicator has to be boolean. Object of type <class 'int'> passed."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLIRM(dml_data_irm, ml_g, LogisticRegression(), normalize_ipw=1)
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLIIVM(dml_data_iivm, ml_g, LogisticRegression(), LogisticRegression(), normalize_ipw=1)
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPQ(dml_data_irm, ml_g, ml_m, treatment=1, normalize_ipw=1)
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLQTE(dml_data_irm, ml_g, ml_m, normalize_ipw=1)
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLLPQ(dml_data_iivm, ml_g, ml_m, treatment=1, normalize_ipw=1)

    # DID models in_sample_normalization
    msg = "in_sample_normalization indicator has to be boolean. Object of type <class 'int'> passed."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLDID(dml_data_did, ml_g, ml_m, in_sample_normalization=1)
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLDIDCS(dml_data_did_cs, ml_g, ml_m, in_sample_normalization=1)


@pytest.mark.ci
def test_doubleml_exception_subgroups():
    msg = 'Invalid subgroups True. subgroups must be of type dictionary.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLIIVM(dml_data_iivm, Lasso(), LogisticRegression(), LogisticRegression(),
                         subgroups=True)
    msg = "Invalid subgroups {'abs': True}. subgroups must be a dictionary with keys always_takers and never_takers."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLIIVM(dml_data_iivm, Lasso(), LogisticRegression(), LogisticRegression(),
                         subgroups={'abs': True})
    msg = ("Invalid subgroups {'always_takers': True, 'never_takers': False, 'abs': 5}. "
           "subgroups must be a dictionary with keys always_takers and never_takers.")
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLIIVM(dml_data_iivm, Lasso(), LogisticRegression(), LogisticRegression(),
                         subgroups={'always_takers': True, 'never_takers': False, 'abs': 5})
    msg = ("Invalid subgroups {'always_takers': True}. "
           "subgroups must be a dictionary with keys always_takers and never_takers.")
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLIIVM(dml_data_iivm, Lasso(), LogisticRegression(), LogisticRegression(),
                         subgroups={'always_takers': True})
    msg = r"subgroups\['always_takers'\] must be True or False. Got 5."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLIIVM(dml_data_iivm, Lasso(), LogisticRegression(), LogisticRegression(),
                         subgroups={'always_takers': 5, 'never_takers': False})
    msg = r"subgroups\['never_takers'\] must be True or False. Got 5."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLIIVM(dml_data_iivm, Lasso(), LogisticRegression(), LogisticRegression(),
                         subgroups={'always_takers': True, 'never_takers': 5})


@pytest.mark.ci
def test_doubleml_exception_resampling():
    msg = "The number of folds must be of int type. 1.5 of type <class 'float'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=1.5)
    msg = ('The number of repetitions for the sample splitting must be of int type. '
           "1.5 of type <class 'float'> was passed.")
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPLR(dml_data, ml_l, ml_m, n_rep=1.5)
    msg = 'The number of folds must be positive. 0 was passed.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=0)
    msg = 'The number of repetitions for the sample splitting must be positive. 0 was passed.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPLR(dml_data, ml_l, ml_m, n_rep=0)
    msg = 'apply_cross_fitting must be True or False. Got 1.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPLR(dml_data, ml_l, ml_m, apply_cross_fitting=1)
    msg = 'draw_sample_splitting must be True or False. Got true.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPLR(dml_data, ml_l, ml_m, draw_sample_splitting='true')


@pytest.mark.ci
def test_doubleml_exception_dml_procedure():
    msg = 'dml_procedure must be "dml1" or "dml2". Got 1.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPLR(dml_data, ml_l, ml_m, dml_procedure='1')
    msg = 'dml_procedure must be "dml1" or "dml2". Got dml.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPLR(dml_data, ml_l, ml_m, dml_procedure='dml')


@pytest.mark.ci
def test_doubleml_warning_crossfitting_onefold():
    msg = 'apply_cross_fitting is set to False. Cross-fitting is not supported for n_folds = 1.'
    with pytest.warns(UserWarning, match=msg):
        _ = DoubleMLPLR(dml_data, ml_l, ml_m, apply_cross_fitting=True, n_folds=1)


@pytest.mark.ci
def test_doubleml_exception_no_cross_fit():
    msg = 'Estimation without cross-fitting not supported for n_folds > 2.'
    with pytest.raises(AssertionError, match=msg):
        _ = DoubleMLPLR(dml_data, ml_l, ml_m, apply_cross_fitting=False)


@pytest.mark.ci
def test_doubleml_exception_get_params():
    msg = 'Invalid nuisance learner ml_r. Valid nuisance learner ml_l or ml_m.'
    with pytest.raises(ValueError, match=msg):
        dml_plr.get_params('ml_r')
    msg = 'Invalid nuisance learner ml_g. Valid nuisance learner ml_l or ml_m.'
    with pytest.raises(ValueError, match=msg):
        dml_plr.get_params('ml_g')
    msg = 'Invalid nuisance learner ml_r. Valid nuisance learner ml_l or ml_m or ml_g.'
    with pytest.raises(ValueError, match=msg):
        dml_plr_iv_type.get_params('ml_r')


@pytest.mark.ci
def test_doubleml_exception_smpls():
    msg = ('Sample splitting not specified. '
           r'Either draw samples via .draw_sample splitting\(\) or set external samples via .set_sample_splitting\(\).')
    dml_plr_no_smpls = DoubleMLPLR(dml_data, ml_l, ml_m, draw_sample_splitting=False)
    with pytest.raises(ValueError, match=msg):
        _ = dml_plr_no_smpls.smpls
    msg = 'Sample splitting not specified. Draw samples via .draw_sample splitting().'
    dml_pliv_cluster_no_smpls = DoubleMLPLIV(dml_cluster_data_pliv, ml_l, ml_m, ml_r, draw_sample_splitting=False)
    with pytest.raises(ValueError, match=msg):
        _ = dml_pliv_cluster_no_smpls.smpls_cluster
    with pytest.raises(ValueError, match=msg):
        _ = dml_pliv_cluster_no_smpls.smpls


@pytest.mark.ci
def test_doubleml_exception_fit():
    msg = "The number of CPUs used to fit the learners must be of int type. 5 of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_plr.fit(n_jobs_cv='5')
    msg = 'store_predictions must be True or False. Got 1.'
    with pytest.raises(TypeError, match=msg):
        dml_plr.fit(store_predictions=1)
    msg = 'store_models must be True or False. Got 1.'
    with pytest.raises(TypeError, match=msg):
        dml_plr.fit(store_models=1)


@pytest.mark.ci
def test_doubleml_exception_bootstrap():
    dml_plr_boot = DoubleMLPLR(dml_data, ml_l, ml_m)
    dml_qte_boot = DoubleMLQTE(dml_data_irm, RandomForestClassifier(), RandomForestClassifier())
    msg = r'Apply fit\(\) before bootstrap\(\).'
    with pytest.raises(ValueError, match=msg):
        dml_plr_boot.bootstrap()
    with pytest.raises(ValueError, match=msg):
        dml_qte_boot.bootstrap()

    dml_plr_boot.fit()
    dml_qte_boot.fit()
    msg = 'Method must be "Bayes", "normal" or "wild". Got Gaussian.'
    with pytest.raises(ValueError, match=msg):
        dml_plr_boot.bootstrap(method='Gaussian')
    with pytest.raises(ValueError, match=msg):
        dml_qte_boot.bootstrap(method='Gaussian')
    msg = "The number of bootstrap replications must be of int type. 500 of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_plr_boot.bootstrap(n_rep_boot='500')
    with pytest.raises(TypeError, match=msg):
        dml_qte_boot.bootstrap(n_rep_boot='500')
    msg = 'The number of bootstrap replications must be positive. 0 was passed.'
    with pytest.raises(ValueError, match=msg):
        dml_plr_boot.bootstrap(n_rep_boot=0)
    with pytest.raises(ValueError, match=msg):
        dml_qte_boot.bootstrap(n_rep_boot=0)


@pytest.mark.ci
def test_doubleml_exception_confint():
    dml_plr_confint = DoubleMLPLR(dml_data, ml_l, ml_m)
    dml_qte_confint = DoubleMLQTE(dml_data_irm, RandomForestClassifier(), RandomForestClassifier())

    msg = 'joint must be True or False. Got 1.'
    with pytest.raises(TypeError, match=msg):
        dml_plr_confint.confint(joint=1)
    with pytest.raises(TypeError, match=msg):
        dml_qte_confint.confint(joint=1)
    msg = "The confidence level must be of float type. 5% of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_plr_confint.confint(level='5%')
    msg = "The confidence level must be of float type. 5% of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_qte_confint.confint(level='5%')
    msg = r'The confidence level must be in \(0,1\). 0.0 was passed.'
    with pytest.raises(ValueError, match=msg):
        dml_plr_confint.confint(level=0.)
    with pytest.raises(ValueError, match=msg):
        dml_qte_confint.confint(level=0.)

    msg = r'Apply fit\(\) before confint\(\).'
    with pytest.raises(ValueError, match=msg):
        dml_plr_confint.confint()
    with pytest.raises(ValueError, match=msg):
        dml_qte_confint.confint()
    msg = r'Apply fit\(\) & bootstrap\(\) before confint\(joint=True\).'
    with pytest.raises(ValueError, match=msg):
        dml_plr_confint.confint(joint=True)
    with pytest.raises(ValueError, match=msg):
        dml_qte_confint.confint(joint=True)
    dml_plr_confint.fit()  # error message should still appear till bootstrap was applied as well
    dml_qte_confint.fit()
    with pytest.raises(ValueError, match=msg):
        dml_plr_confint.confint(joint=True)
    with pytest.raises(ValueError, match=msg):
        dml_qte_confint.confint(joint=True)
    dml_plr_confint.bootstrap()
    dml_qte_confint.bootstrap()
    df_plr_ci = dml_plr_confint.confint(joint=True)
    df_qte_ci = dml_qte_confint.confint(joint=True)
    assert isinstance(df_plr_ci, pd.DataFrame)
    assert isinstance(df_qte_ci, pd.DataFrame)


@pytest.mark.ci
def test_doubleml_exception_p_adjust():
    dml_plr_p_adjust = DoubleMLPLR(dml_data, ml_l, ml_m)

    msg = r'Apply fit\(\) before p_adjust\(\).'
    with pytest.raises(ValueError, match=msg):
        dml_plr_p_adjust.p_adjust()
    dml_plr_p_adjust.fit()
    msg = r'Apply fit\(\) & bootstrap\(\) before p_adjust'
    with pytest.raises(ValueError, match=msg):
        dml_plr_p_adjust.p_adjust(method='romano-wolf')
    dml_plr_p_adjust.bootstrap()
    p_val = dml_plr_p_adjust.p_adjust(method='romano-wolf')
    assert isinstance(p_val, pd.DataFrame)

    msg = "The p_adjust method must be of str type. 0.05 of type <class 'float'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_plr_p_adjust.p_adjust(method=0.05)


@pytest.mark.ci
def test_doubleml_exception_tune():
    msg = r'Invalid param_grids \[0.05, 0.5\]. param_grids must be a dictionary with keys ml_l and ml_m'
    with pytest.raises(ValueError, match=msg):
        dml_plr.tune([0.05, 0.5])
    msg = (r"Invalid param_grids {'ml_r': {'alpha': \[0.05, 0.5\]}}. "
           "param_grids must be a dictionary with keys ml_l and ml_m.")
    with pytest.raises(ValueError, match=msg):
        dml_plr.tune({'ml_r': {'alpha': [0.05, 0.5]}})

    msg = r'Invalid param_grids \[0.05, 0.5\]. param_grids must be a dictionary with keys ml_l and ml_m and ml_g'
    with pytest.raises(ValueError, match=msg):
        dml_plr_iv_type.tune([0.05, 0.5])
    msg = (r"Invalid param_grids {'ml_g': {'alpha': \[0.05, 0.5\]}, 'ml_m': {'alpha': \[0.05, 0.5\]}}. "
           "param_grids must be a dictionary with keys ml_l and ml_m and ml_g.")
    with pytest.raises(ValueError, match=msg):
        dml_plr_iv_type.tune({'ml_g': {'alpha': [0.05, 0.5]},
                              'ml_m': {'alpha': [0.05, 0.5]}})

    msg = 'Learner ml_g was renamed to ml_l. '
    with pytest.warns(DeprecationWarning, match=msg):
        dml_plr.tune({'ml_g': {'alpha': [0.05, 0.5]},
                      'ml_m': {'alpha': [0.05, 0.5]}})
    with pytest.warns(DeprecationWarning, match=msg):
        dml_plr.tune({'ml_l': {'alpha': [0.05, 0.5]},
                      'ml_m': {'alpha': [0.05, 0.5]}},
                     scoring_methods={'ml_g': 'explained_variance',
                                      'ml_m': 'explained_variance'})

    msg = 'Learner ml_g was renamed to ml_l. '
    with pytest.warns(DeprecationWarning, match=msg):
        dml_pliv.tune({'ml_g': {'alpha': [0.05, 0.5]},
                       'ml_m': {'alpha': [0.05, 0.5]},
                       'ml_r': {'alpha': [0.05, 0.5]}})
    with pytest.warns(DeprecationWarning, match=msg):
        dml_pliv.tune({'ml_l': {'alpha': [0.05, 0.5]},
                       'ml_m': {'alpha': [0.05, 0.5]},
                       'ml_r': {'alpha': [0.05, 0.5]}},
                      scoring_methods={'ml_g': 'explained_variance',
                                       'ml_m': 'explained_variance',
                                       'ml_r': 'explained_variance'})

    param_grids = {'ml_l': {'alpha': [0.05, 0.5]}, 'ml_m': {'alpha': [0.05, 0.5]}}
    msg = ('Invalid scoring_methods neg_mean_absolute_error. '
           'scoring_methods must be a dictionary. '
           'Valid keys are ml_l and ml_m.')
    with pytest.raises(ValueError, match=msg):
        dml_plr.tune(param_grids, scoring_methods='neg_mean_absolute_error')

    msg = 'tune_on_folds must be True or False. Got 1.'
    with pytest.raises(TypeError, match=msg):
        dml_plr.tune(param_grids, tune_on_folds=1)

    msg = 'The number of folds used for tuning must be at least two. 1 was passed.'
    with pytest.raises(ValueError, match=msg):
        dml_plr.tune(param_grids, n_folds_tune=1)
    msg = "The number of folds used for tuning must be of int type. 1.0 of type <class 'float'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_plr.tune(param_grids, n_folds_tune=1.)

    msg = 'search_mode must be "grid_search" or "randomized_search". Got gridsearch.'
    with pytest.raises(ValueError, match=msg):
        dml_plr.tune(param_grids, search_mode='gridsearch')

    msg = 'The number of parameter settings sampled for the randomized search must be at least two. 1 was passed.'
    with pytest.raises(ValueError, match=msg):
        dml_plr.tune(param_grids, n_iter_randomized_search=1)
    msg = ("The number of parameter settings sampled for the randomized search must be of int type. "
           "1.0 of type <class 'float'> was passed.")
    with pytest.raises(TypeError, match=msg):
        dml_plr.tune(param_grids, n_iter_randomized_search=1.)

    msg = "The number of CPUs used to fit the learners must be of int type. 5 of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_plr.tune(param_grids, n_jobs_cv='5')

    msg = 'set_as_params must be True or False. Got 1.'
    with pytest.raises(TypeError, match=msg):
        dml_plr.tune(param_grids, set_as_params=1)

    msg = 'return_tune_res must be True or False. Got 1.'
    with pytest.raises(TypeError, match=msg):
        dml_plr.tune(param_grids, return_tune_res=1)


@pytest.mark.ci
def test_doubleml_exception_set_ml_nuisance_params():
    msg = 'Invalid nuisance learner g. Valid nuisance learner ml_l or ml_m.'
    with pytest.raises(ValueError, match=msg):
        dml_plr.set_ml_nuisance_params('g', 'd', {'alpha': 0.1})
    msg = 'Invalid treatment variable y. Valid treatment variable d.'
    with pytest.raises(ValueError, match=msg):
        dml_plr.set_ml_nuisance_params('ml_l', 'y', {'alpha': 0.1})


class _DummyNoSetParams:
    def fit(self):
        pass


class _DummyNoGetParams(_DummyNoSetParams):
    def set_params(self):
        pass


class _DummyNoClassifier(_DummyNoGetParams):
    def get_params(self):
        pass

    def predict_proba(self):
        pass


class LogisticRegressionManipulatedPredict(LogisticRegression):
    def predict(self, X):
        if self.max_iter == 314:
            preds = super().predict_proba(X)[:, 1]
        else:
            preds = super().predict(X)
        return preds


@pytest.mark.ci
def test_doubleml_exception_learner():
    err_msg_prefix = 'Invalid learner provided for ml_l: '
    warn_msg_prefix = 'Learner provided for ml_l is probably invalid: '

    msg = err_msg_prefix + 'provide an instance of a learner instead of a class.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPLR(dml_data, Lasso, ml_m)
    msg = err_msg_prefix + r'BaseEstimator\(\) has no method .fit\(\).'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPLR(dml_data, BaseEstimator(), ml_m)
    # msg = err_msg_prefix + r'_DummyNoSetParams\(\) has no method .set_params\(\).'
    with pytest.raises(TypeError):
        _ = DoubleMLPLR(dml_data, _DummyNoSetParams(), ml_m)
    # msg = err_msg_prefix + r'_DummyNoSetParams\(\) has no method .get_params\(\).'
    with pytest.raises(TypeError):
        _ = DoubleMLPLR(dml_data, _DummyNoGetParams(), ml_m)

    # msg = 'Learner provided for ml_m is probably invalid: ' + r'_DummyNoClassifier\(\) is \(probably\) no classifier.'
    with pytest.warns(UserWarning):
        _ = DoubleMLIRM(dml_data_irm, Lasso(), _DummyNoClassifier())

    # ToDo: Currently for ml_l (and others) we only check whether the learner can be identified as regressor. However,
    # we do not check whether it can instead be identified as classifier, which could be used to throw an error.
    msg = warn_msg_prefix + r'LogisticRegression\(\) is \(probably\) no regressor.'
    with pytest.warns(UserWarning, match=msg):
        _ = DoubleMLPLR(dml_data, LogisticRegression(), Lasso())

    # we allow classifiers for ml_m in PLR, but only for binary treatment variables
    msg = (r'The ml_m learner LogisticRegression\(\) was identified as classifier '
           'but at least one treatment variable is not binary with values 0 and 1.')
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPLR(dml_data, Lasso(), LogisticRegression())

    msg = r"For score = 'IV-type', learners ml_l and ml_g should be specified. Set ml_g = clone\(ml_l\)."
    with pytest.warns(UserWarning, match=msg):
        _ = DoubleMLPLR(dml_data, ml_l=Lasso(), ml_m=ml_m, score='IV-type')

    msg = 'A learner ml_g has been provided for score = "partialling out" but will be ignored.'
    with pytest.warns(UserWarning, match=msg):
        _ = DoubleMLPLR(dml_data, ml_l=Lasso(), ml_m=Lasso(), ml_g=Lasso(), score='partialling out')

    msg = "For score = 'IV-type', learners ml_l, ml_m, ml_r and ml_g need to be specified."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPLIV(dml_data_pliv, ml_l=ml_l, ml_m=ml_m, ml_r=ml_r,
                         score='IV-type')

    msg = 'A learner ml_g has been provided for score = "partialling out" but will be ignored.'
    with pytest.warns(UserWarning, match=msg):
        _ = DoubleMLPLIV(dml_data_pliv, ml_l=Lasso(), ml_m=Lasso(), ml_r=Lasso(), ml_g=Lasso(), score='partialling out')

    # we allow classifiers for ml_g for binary treatment variables in IRM
    msg = (r'The ml_g learner LogisticRegression\(\) was identified as classifier '
           'but the outcome variable is not binary with values 0 and 1.')
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLIRM(dml_data_irm, LogisticRegression(), LogisticRegression())

    # we allow classifiers for ml_g for binary treatment variables in IRM
    msg = (r'The ml_g learner LogisticRegression\(\) was identified as classifier '
           'but the outcome variable is not binary with values 0 and 1.')
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLIIVM(dml_data_iivm, LogisticRegression(), LogisticRegression(), LogisticRegression())

    # construct a classifier which is not identifiable as classifier via is_classifier by sklearn
    # it then predicts labels and therefore an exception will be thrown
    log_reg = LogisticRegression()
    log_reg._estimator_type = None
    msg = (r'Learner provided for ml_m is probably invalid: LogisticRegression\(\) is \(probably\) neither a regressor '
           'nor a classifier. Method predict is used for prediction.')
    with pytest.warns(UserWarning, match=msg):
        dml_plr_hidden_classifier = DoubleMLPLR(dml_data_irm, Lasso(), log_reg)
    msg = (r'For the binary treatment variable d, predictions obtained with the ml_m learner LogisticRegression\(\) '
           'are also observed to be binary with values 0 and 1. Make sure that for classifiers probabilities and not '
           'labels are predicted.')
    with pytest.raises(ValueError, match=msg):
        dml_plr_hidden_classifier.fit()

    # construct a classifier which is not identifiable as classifier via is_classifier by sklearn
    # it then predicts labels and therefore an exception will be thrown
    # whether predict() or predict_proba() is being called can also be manipulated via the unrelated max_iter variable
    log_reg = LogisticRegressionManipulatedPredict()
    log_reg._estimator_type = None
    msg = (r'Learner provided for ml_g is probably invalid: LogisticRegressionManipulatedPredict\(\) is \(probably\) '
           'neither a regressor nor a classifier. Method predict is used for prediction.')
    with pytest.warns(UserWarning, match=msg):
        dml_irm_hidden_classifier = DoubleMLIRM(dml_data_irm_binary_outcome,
                                                log_reg, LogisticRegression())
    msg = (r'For the binary outcome variable y, predictions obtained with the ml_g learner '
           r'LogisticRegressionManipulatedPredict\(\) are also observed to be binary with values 0 and 1. Make sure '
           'that for classifiers probabilities and not labels are predicted.')
    with pytest.raises(ValueError, match=msg):
        dml_irm_hidden_classifier.fit()
    with pytest.raises(ValueError, match=msg):
        dml_irm_hidden_classifier.set_ml_nuisance_params('ml_g0', 'd', {'max_iter': 314})
        dml_irm_hidden_classifier.fit()

    msg = (r'Learner provided for ml_g is probably invalid: LogisticRegressionManipulatedPredict\(\) is \(probably\) '
           'neither a regressor nor a classifier. Method predict is used for prediction.')
    with pytest.warns(UserWarning, match=msg):
        dml_iivm_hidden_classifier = DoubleMLIIVM(dml_data_iivm_binary_outcome,
                                                  log_reg, LogisticRegression(), LogisticRegression())
    msg = (r'For the binary outcome variable y, predictions obtained with the ml_g learner '
           r'LogisticRegressionManipulatedPredict\(\) are also observed to be binary with values 0 and 1. Make sure '
           'that for classifiers probabilities and not labels are predicted.')
    with pytest.raises(ValueError, match=msg):
        dml_iivm_hidden_classifier.fit()
    with pytest.raises(ValueError, match=msg):
        dml_iivm_hidden_classifier.set_ml_nuisance_params('ml_g0', 'd', {'max_iter': 314})
        dml_iivm_hidden_classifier.fit()

    msg = 'Learner ml_g was renamed to ml_l. '
    with pytest.warns(DeprecationWarning, match=msg):
        dml_plr.set_ml_nuisance_params('ml_g', 'd', {'max_iter': 314})

    msg = 'Learner ml_g was renamed to ml_l. '
    with pytest.warns(DeprecationWarning, match=msg):
        dml_pliv.set_ml_nuisance_params('ml_g', 'd', {'max_iter': 314})


@pytest.mark.ci
@pytest.mark.filterwarnings("ignore:Learner provided for")
def test_doubleml_exception_and_warning_learner():
    # msg = err_msg_prefix + r'_DummyNoClassifier\(\) has no method .predict\(\).'
    with pytest.raises(TypeError):
        _ = DoubleMLPLR(dml_data, _DummyNoClassifier(), Lasso())
    msg = 'Invalid learner provided for ml_m: ' + r'Lasso\(\) has no method .predict_proba\(\).'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLIRM(dml_data_irm, Lasso(), Lasso())


@pytest.mark.ci
def test_doubleml_cluster_not_yet_implemented():
    dml_pliv_cluster = DoubleMLPLIV(dml_cluster_data_pliv, ml_g, ml_m, ml_r)
    dml_pliv_cluster.fit()
    msg = 'bootstrap not yet implemented with clustering.'
    with pytest.raises(NotImplementedError, match=msg):
        _ = dml_pliv_cluster.bootstrap()

    smpls = dml_plr.smpls
    msg = ('Externally setting the sample splitting for DoubleML is '
           'not yet implemented with clustering.')
    with pytest.raises(NotImplementedError, match=msg):
        _ = dml_pliv_cluster.set_sample_splitting(smpls)

    df = dml_cluster_data_pliv.data.copy()
    df['cluster_var_k'] = df['cluster_var_i'] + df['cluster_var_j'] - 2
    dml_cluster_data_multiway = DoubleMLClusterData(df, y_col='Y', d_cols='D', x_cols=['X1', 'X5'], z_cols='Z',
                                                    cluster_cols=['cluster_var_i', 'cluster_var_j', 'cluster_var_k'])
    assert dml_cluster_data_multiway.n_cluster_vars == 3
    msg = r'Multi-way \(n_ways > 2\) clustering not yet implemented.'
    with pytest.raises(NotImplementedError, match=msg):
        _ = DoubleMLPLIV(dml_cluster_data_multiway, ml_g, ml_m, ml_r)

    msg = (r'No cross-fitting \(`apply_cross_fitting = False`\) '
           'is not yet implemented with clustering.')
    with pytest.raises(NotImplementedError, match=msg):
        _ = DoubleMLPLIV(dml_cluster_data_pliv, ml_g, ml_m, ml_r,
                         n_folds=1)
    with pytest.raises(NotImplementedError, match=msg):
        _ = DoubleMLPLIV(dml_cluster_data_pliv, ml_g, ml_m, ml_r,
                         apply_cross_fitting=False, n_folds=2)


class LassoWithNanPred(Lasso):
    def predict(self, X):
        preds = super().predict(X)
        n_obs = len(preds)
        preds[np.random.randint(0, n_obs, 1)] = np.nan
        return preds


class LassoWithInfPred(Lasso):
    def predict(self, X):
        preds = super().predict(X)
        n_obs = len(preds)
        preds[np.random.randint(0, n_obs, 1)] = np.inf
        return preds


@pytest.mark.ci
def test_doubleml_nan_prediction():
    msg = r'Predictions from learner LassoWithNanPred\(\) for ml_l are not finite.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPLR(dml_data, LassoWithNanPred(), ml_m).fit()
    msg = r'Predictions from learner LassoWithInfPred\(\) for ml_l are not finite.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPLR(dml_data, LassoWithInfPred(), ml_m).fit()

    msg = r'Predictions from learner LassoWithNanPred\(\) for ml_m are not finite.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPLR(dml_data, ml_l, LassoWithNanPred()).fit()
    msg = r'Predictions from learner LassoWithInfPred\(\) for ml_m are not finite.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPLR(dml_data, ml_l, LassoWithInfPred()).fit()


@pytest.mark.ci
def test_doubleml_exception_gate():
    dml_irm_obj = DoubleMLIRM(dml_data_irm,
                              ml_g=Lasso(),
                              ml_m=LogisticRegression(),
                              trimming_threshold=0.05,
                              n_folds=5)
    dml_irm_obj.fit()

    msg = "Groups must be of DataFrame type. Groups of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_irm_obj.gate(groups=2)
    msg = (r'Columns of groups must be of bool type or int type \(dummy coded\). '
           'Alternatively, groups should only contain one column.')
    with pytest.raises(TypeError, match=msg):
        dml_irm_obj.gate(groups=pd.DataFrame(np.random.normal(0, 1, size=(dml_data_irm.n_obs, 3))))

    dml_irm_obj = DoubleMLIRM(dml_data_irm,
                              ml_g=Lasso(),
                              ml_m=LogisticRegression(),
                              trimming_threshold=0.05,
                              n_folds=5,
                              score='ATTE')
    dml_irm_obj.fit()

    msg = 'Invalid score ATTE. Valid score ATE.'
    with pytest.raises(ValueError, match=msg):
        dml_irm_obj.gate(groups=2)

    dml_irm_obj = DoubleMLIRM(dml_data_irm,
                              ml_g=Lasso(),
                              ml_m=LogisticRegression(),
                              trimming_threshold=0.05,
                              n_folds=5,
                              score='ATE',
                              n_rep=2)
    dml_irm_obj.fit()

    msg = 'Only implemented for one repetition. Number of repetitions is 2.'
    with pytest.raises(NotImplementedError, match=msg):
        dml_irm_obj.gate(groups=2)


@pytest.mark.ci
def test_doubleml_exception_cate():
    dml_irm_obj = DoubleMLIRM(dml_data_irm,
                              ml_g=Lasso(),
                              ml_m=LogisticRegression(),
                              trimming_threshold=0.05,
                              n_folds=5,
                              score='ATTE')
    dml_irm_obj.fit()

    msg = 'Invalid score ATTE. Valid score ATE.'
    with pytest.raises(ValueError, match=msg):
        dml_irm_obj.cate(basis=2)

    dml_irm_obj = DoubleMLIRM(dml_data_irm,
                              ml_g=Lasso(),
                              ml_m=LogisticRegression(),
                              trimming_threshold=0.05,
                              n_folds=5,
                              score='ATE',
                              n_rep=2)
    dml_irm_obj.fit()
    msg = 'Only implemented for one repetition. Number of repetitions is 2.'
    with pytest.raises(NotImplementedError, match=msg):
        dml_irm_obj.cate(basis=2)


@pytest.mark.ci
def test_double_ml_exception_evaluate_learner():
    dml_irm_obj = DoubleMLIRM(dml_data_irm,
                              ml_g=Lasso(),
                              ml_m=LogisticRegression(),
                              trimming_threshold=0.05,
                              n_folds=5,
                              score='ATTE')

    msg = r'Apply fit\(\) before evaluate_learners\(\).'
    with pytest.raises(ValueError, match=msg):
        dml_irm_obj.evaluate_learners()

    dml_irm_obj.fit()

    msg = "metric should be a callable. 'mse' was passed."
    with pytest.raises(TypeError, match=msg):
        dml_irm_obj.evaluate_learners(metric="mse")

    msg = (r"The learners have to be a subset of \['ml_g0', 'ml_g1', 'ml_m'\]. "
           r"Learners \['ml_g', 'ml_m'\] provided.")
    with pytest.raises(ValueError, match=msg):
        dml_irm_obj.evaluate_learners(learners=['ml_g', 'ml_m'])

    msg = 'Evaluation from learner ml_g0 is not finite.'

    def eval_fct(y_pred, y_true):
        return np.nan
    with pytest.raises(ValueError, match=msg):
        dml_irm_obj.evaluate_learners(metric=eval_fct)
