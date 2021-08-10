import pytest
import numpy as np

from doubleml import DoubleMLPLR
from doubleml.datasets import make_plr_CCDDHNR2018

from sklearn.linear_model import Lasso

np.random.seed(3141)
dml_data = make_plr_CCDDHNR2018(n_obs=10)
ml_g = Lasso()
ml_m = Lasso()
dml_plr = DoubleMLPLR(dml_data, ml_g, ml_m,
                      n_folds=7, n_rep=8,
                      draw_sample_splitting=False)


def _assert_resampling_pars(dml_obj0, dml_obj1):
    assert dml_obj0.n_folds == dml_obj1.n_folds
    assert dml_obj0.n_rep == dml_obj1.n_rep
    assert dml_obj0.apply_cross_fitting == dml_obj1.apply_cross_fitting
    _assert_smpls_equal(dml_obj0.smpls, dml_obj1.smpls, dml_obj0.apply_cross_fitting)


def _assert_smpls_equal(smpls0, smpls1, apply_cross_fitting=True):
    assert len(smpls0) == len(smpls1)
    for i_rep, _ in enumerate(smpls0):
        assert len(smpls0[i_rep]) == len(smpls1[i_rep])
        if apply_cross_fitting:
            for i_fold, _ in enumerate(smpls0[i_rep]):
                assert np.array_equal(smpls0[i_rep][i_fold][0], smpls1[i_rep][i_fold][0])
                assert np.array_equal(smpls0[i_rep][i_fold][1], smpls1[i_rep][i_fold][1])
        else:
            assert len(smpls0[i_rep]) == 1
            assert np.array_equal(smpls0[i_rep][0][0], smpls1[i_rep][0][0])
            assert np.array_equal(smpls0[i_rep][0][1], smpls1[i_rep][0][1])


@pytest.mark.ci
def test_doubleml_set_sample_splitting_tuple():
    # simple sample splitting with two folds and without cross-fitting
    smpls = ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])
    dml_plr.set_sample_splitting(smpls)

    assert dml_plr.n_folds == 2
    assert dml_plr.n_rep == 1
    assert not dml_plr.apply_cross_fitting
    _assert_smpls_equal([[smpls]], dml_plr.smpls, apply_cross_fitting=dml_plr.apply_cross_fitting)

    # no cross-fitting, no sample-splitting
    smpls = ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    dml_plr.set_sample_splitting(smpls)
    assert dml_plr.n_folds == 1
    assert dml_plr.n_rep == 1
    assert not dml_plr.apply_cross_fitting
    _assert_smpls_equal([[smpls]], dml_plr.smpls, apply_cross_fitting=dml_plr.apply_cross_fitting)

    smpls = ([0, 1, 2, 3, 4], [5, 6], [7, 8, 9])
    msg = 'Invalid partition provided. ' + 'Tuple for train_ind and test_ind must consist of exactly two elements.'
    with pytest.raises(ValueError, match=msg):
        dml_plr.set_sample_splitting(smpls)


@pytest.mark.ci
def test_doubleml_set_sample_splitting_all_tuple():
    # sample splitting with two folds and cross-fitting but no repeated cross-fitting
    smpls = [([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]),
             ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])]
    dml_plr.set_sample_splitting(smpls)
    assert dml_plr.n_folds == 2
    assert dml_plr.n_rep == 1
    assert dml_plr.apply_cross_fitting
    _assert_smpls_equal([smpls], dml_plr.smpls, apply_cross_fitting=dml_plr.apply_cross_fitting)

    smpls = [([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]),
             ([5, 6, 7, 8, 9], [0, 1, 2], [3, 4])]
    msg = 'Invalid partition provided. ' + 'All tuples for train_ind and test_ind must consist of exactly two elements.'
    with pytest.raises(ValueError, match=msg):
        dml_plr.set_sample_splitting(smpls)

    # simple sample splitting with two folds and without cross-fitting
    smpls = [([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])]
    dml_plr.set_sample_splitting(smpls)

    assert dml_plr.n_folds == 2
    assert dml_plr.n_rep == 1
    assert not dml_plr.apply_cross_fitting
    _assert_smpls_equal([smpls], dml_plr.smpls, apply_cross_fitting=dml_plr.apply_cross_fitting)

    # sample splitting with cross-fitting and two folds that do not form a partition
    smpls = [([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]),
             ([5, 6, 7, 8], [0, 1, 2, 3, 4, 9])]
    msg = 'Invalid partition provided. ' + 'Tuples for more than one fold provided that don\'t form a partition.'
    with pytest.raises(ValueError, match=msg):
        dml_plr.set_sample_splitting(smpls)


@pytest.mark.ci
def test_doubleml_set_sample_splitting_all_list():
    # sample splitting with two folds and repeated cross-fitting with n_rep = 2
    smpls = [[([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]),
              ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])],
             [([0, 2, 4, 6, 8], [1, 3, 5, 7, 9]),
              ([1, 3, 5, 7, 9], [0, 2, 4, 6, 8])]]
    dml_plr.set_sample_splitting(smpls)
    assert dml_plr.n_folds == 2
    assert dml_plr.n_rep == 2
    assert dml_plr.apply_cross_fitting
    _assert_smpls_equal(smpls, dml_plr.smpls, apply_cross_fitting=dml_plr.apply_cross_fitting)

    smpls = np.array(([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]))
    # msg = ("all_smpls must be of list or tuple type. [[0 1 2 3 4]"
    #        r"\n [5 6 7 8 9]] of type <class 'numpy.ndarray'> was passed.")
    with pytest.raises(TypeError):
        dml_plr.set_sample_splitting(smpls)

    # second sample splitting is not a list
    smpls = [[([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]),
              ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])],
             (([0, 2, 4, 6, 8], [1, 3, 5, 7, 9]),
              ([1, 3, 5, 7, 9], [0, 2, 4, 6, 8]))]
    msg = ('Invalid partition provided. '
           'all_smpls is a list where neither all elements are tuples nor all elements are lists.')
    with pytest.raises(ValueError, match=msg):
        dml_plr.set_sample_splitting(smpls)

    smpls = [[([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]),
              ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])],
             [[[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]],  # not a tuple
              ([1, 3, 5, 7, 9], [0, 2, 4, 6, 8])]]
    msg = 'For repeated sample splitting all_smpls must be list of lists of tuples.'
    with pytest.raises(TypeError, match=msg):
        dml_plr.set_sample_splitting(smpls)

    smpls = [[([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]),
              ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])],
             [([0, 2, 4, 6, 8], [1, 3, 5, 7, 9]),
              ([1, 3, 5, 7, 9], [0, 2, 4], [6, 8])]]
    msg = 'Invalid partition provided. ' + 'All tuples for train_ind and test_ind must consist of exactly two elements.'
    with pytest.raises(ValueError, match=msg):
        dml_plr.set_sample_splitting(smpls)

    smpls = [[([0, 1, 2, 3, 6], [4, 5, 7, 8, 9]),
              ([4, 5, 7, 8, 9], [0, 1, 2, 3, 6])],
             [([0, 1, 4, 5, 7, 9], [2, 3, 6, 8]),
              ([0, 2, 3, 4, 6, 8, 9], [1, 5, 7]),
              ([1, 2, 3, 5, 6, 7, 8], [0, 4, 9])]]
    msg = 'Invalid partition provided. ' + 'Different number of folds for repeated sample splitting.'
    with pytest.raises(ValueError, match=msg):
        dml_plr.set_sample_splitting(smpls)

    # sample splitting with cross-fitting and two folds that do not form a partition
    smpls = [[([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]),
              ([5, 6, 7, 8], [0, 1, 2, 3, 4, 9])]]
    msg = ('Invalid partition provided. '
           'Tuples for more than one fold provided '
           'but at least one does not form a partition.')
    with pytest.raises(ValueError, match=msg):
        dml_plr.set_sample_splitting(smpls)

    # repeated no-cross-fitting
    smpls = [[([0, 1, 5, 7, 9], [2, 3, 4, 6, 8])],
             [([2, 4, 7, 8, 9], [0, 1, 3, 5, 6])],
             [([0, 1, 4, 6, 8], [2, 3, 5, 7, 9])]]
    dml_plr.set_sample_splitting(smpls)
    assert dml_plr.n_folds == 2
    assert dml_plr.n_rep == 3
    assert not dml_plr.apply_cross_fitting
    _assert_smpls_equal(smpls, dml_plr.smpls, apply_cross_fitting=dml_plr.apply_cross_fitting)


@pytest.mark.ci
def test_doubleml_draw_vs_set():
    np.random.seed(3141)
    dml_plr_set = DoubleMLPLR(dml_data, ml_g, ml_m, n_folds=7, n_rep=8)

    dml_plr_drawn = DoubleMLPLR(dml_data, ml_g, ml_m,
                                n_folds=1, n_rep=1, apply_cross_fitting=False)
    dml_plr_set.set_sample_splitting(dml_plr_drawn.smpls)
    _assert_resampling_pars(dml_plr_drawn, dml_plr_set)
    dml_plr_set.set_sample_splitting(dml_plr_drawn.smpls[0])
    _assert_resampling_pars(dml_plr_drawn, dml_plr_set)
    dml_plr_set.set_sample_splitting(dml_plr_drawn.smpls[0][0])
    _assert_resampling_pars(dml_plr_drawn, dml_plr_set)

    dml_plr_drawn = DoubleMLPLR(dml_data, ml_g, ml_m,
                                n_folds=2, n_rep=1, apply_cross_fitting=False)
    dml_plr_set.set_sample_splitting(dml_plr_drawn.smpls)
    _assert_resampling_pars(dml_plr_drawn, dml_plr_set)
    dml_plr_set.set_sample_splitting(dml_plr_drawn.smpls[0])
    _assert_resampling_pars(dml_plr_drawn, dml_plr_set)
    dml_plr_set.set_sample_splitting(dml_plr_drawn.smpls[0][0])
    _assert_resampling_pars(dml_plr_drawn, dml_plr_set)

    dml_plr_drawn = DoubleMLPLR(dml_data, ml_g, ml_m,
                                n_folds=2, n_rep=1, apply_cross_fitting=True)
    dml_plr_set.set_sample_splitting(dml_plr_drawn.smpls)
    _assert_resampling_pars(dml_plr_drawn, dml_plr_set)
    dml_plr_set.set_sample_splitting(dml_plr_drawn.smpls[0])
    _assert_resampling_pars(dml_plr_drawn, dml_plr_set)

    dml_plr_drawn = DoubleMLPLR(dml_data, ml_g, ml_m,
                                n_folds=5, n_rep=1, apply_cross_fitting=True)
    dml_plr_set.set_sample_splitting(dml_plr_drawn.smpls)
    _assert_resampling_pars(dml_plr_drawn, dml_plr_set)
    dml_plr_set.set_sample_splitting(dml_plr_drawn.smpls[0])
    _assert_resampling_pars(dml_plr_drawn, dml_plr_set)

    dml_plr_drawn = DoubleMLPLR(dml_data, ml_g, ml_m,
                                n_folds=5, n_rep=3, apply_cross_fitting=True)
    dml_plr_set.set_sample_splitting(dml_plr_drawn.smpls)
    _assert_resampling_pars(dml_plr_drawn, dml_plr_set)

    dml_plr_drawn = DoubleMLPLR(dml_data, ml_g, ml_m,
                                n_folds=2, n_rep=4, apply_cross_fitting=False)
    dml_plr_set.set_sample_splitting(dml_plr_drawn.smpls)
    _assert_resampling_pars(dml_plr_drawn, dml_plr_set)


@pytest.mark.ci
def test_doubleml_set_sample_splitting_invalid_sets():
    # sample splitting with two folds and repeated cross-fitting with n_rep = 2
    smpls = [[([0, 1.2, 2, 3, 4], [5, 6, 7, 8, 9]),
              ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])],
             [([0, 2, 4, 6, 8], [1, 3, 5, 7, 9]),
              ([1, 3, 5, 7, 9], [0, 2, 4, 6, 8])]]
    msg = 'Invalid sample split. Train indices must be of type integer.'
    with pytest.raises(TypeError, match=msg):
        dml_plr.set_sample_splitting(smpls)

    smpls = [[([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]),
              ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])],
             [([0, 2, 4, 6, 8], [1, 3.5, 5, 7, 9]),
              ([1, 3, 5, 7, 9], [0, 2, 4, 6, 8])]]
    msg = 'Invalid sample split. Test indices must be of type integer.'
    with pytest.raises(TypeError, match=msg):
        dml_plr.set_sample_splitting(smpls)

    smpls = [[([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]),
              ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])],
             [([0, 2, 3, 4, 6, 8], [1, 3, 5, 7, 9]),
              ([1, 5, 7, 9], [0, 2, 4, 6, 8])]]
    msg = 'Invalid sample split. Intersection of train and test indices is not empty.'
    with pytest.raises(ValueError, match=msg):
        dml_plr.set_sample_splitting(smpls)

    smpls = [[([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]),
              ([5, 6, 7, 7, 8, 9], [0, 1, 2, 3, 4])],
             [([0, 2, 4, 4, 6, 8], [1, 3, 5, 7, 9]),
              ([1, 3, 5, 7, 9], [0, 2, 4, 6, 8])]]
    msg = 'Invalid sample split. Train indices contain non-unique entries.'
    with pytest.raises(ValueError, match=msg):
        dml_plr.set_sample_splitting(smpls)

    smpls = [[([0, 1, 2, 3, 4], [5, 5, 6, 7, 8, 9]),
              ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])],
             [([0, 2, 4, 6, 8], [1, 3, 5, 7, 9]),
              ([1, 3, 5, 7, 9], [0, 2, 4, 6, 8])]]
    msg = 'Invalid sample split. Test indices contain non-unique entries.'
    with pytest.raises(ValueError, match=msg):
        dml_plr.set_sample_splitting(smpls)

    smpls = [[([0, 1, 2, 3, 20], [5, 6, 7, 8, 9]),
              ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])],
             [([0, 2, 4, 6, 8], [1, 3, 5, 7, 9]),
              ([1, 3, 5, 7, 9], [0, 2, 4, 6, 8])]]
    msg = r'Invalid sample split. Train indices must be in \[0, n_obs\).'
    with pytest.raises(ValueError, match=msg):
        dml_plr.set_sample_splitting(smpls)

    smpls = [[([0, 1, 2, 3, 4], [5, -6, 7, 8, 9]),
              ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])],
             [([0, 2, 4, 6, 8], [1, 3, 5, 7, 9]),
              ([1, 3, 5, 7, 9], [0, 2, 4, 6, 8])]]
    msg = r'Invalid sample split. Test indices must be in \[0, n_obs\).'
    with pytest.raises(ValueError, match=msg):
        dml_plr.set_sample_splitting(smpls)
