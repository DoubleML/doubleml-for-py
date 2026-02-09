"""Test sample splitting setup for scalar DoubleML models."""

import numpy as np
import pytest

from doubleml.plm.datasets import make_plr_CCDDHNR2018
from doubleml.plm.plr_scalar import PLR


def _assert_smpls_equal(smpls0, smpls1):
    assert len(smpls0) == len(smpls1)
    for i_rep in range(len(smpls0)):
        assert len(smpls0[i_rep]) == len(smpls1[i_rep])
        for i_fold in range(len(smpls0[i_rep])):
            assert np.array_equal(smpls0[i_rep][i_fold][0], smpls1[i_rep][i_fold][0])
            assert np.array_equal(smpls0[i_rep][i_fold][1], smpls1[i_rep][i_fold][1])


@pytest.mark.ci
def test_scalar_set_sample_splitting_list():
    """Ensure list-of-tuples splits set n_folds/n_rep correctly."""
    np.random.seed(3141)
    dml_data = make_plr_CCDDHNR2018(n_obs=10)
    dml_obj = PLR(dml_data)

    smpls = [([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]), ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])]
    dml_obj.set_sample_splitting(smpls)

    assert dml_obj.n_folds == 2
    assert dml_obj.n_rep == 1
    _assert_smpls_equal([smpls], dml_obj.smpls)


@pytest.mark.ci
def test_scalar_set_sample_splitting_list_of_lists():
    """Ensure list-of-list splits set repeated sample splitting correctly."""
    np.random.seed(3141)
    dml_data = make_plr_CCDDHNR2018(n_obs=10)
    dml_obj = PLR(dml_data)

    smpls = [
        [([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]), ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])],
        [([0, 2, 4, 6, 8], [1, 3, 5, 7, 9]), ([1, 3, 5, 7, 9], [0, 2, 4, 6, 8])],
    ]
    dml_obj.set_sample_splitting(smpls)

    assert dml_obj.n_folds == 2
    assert dml_obj.n_rep == 2
    _assert_smpls_equal(smpls, dml_obj.smpls)


@pytest.mark.ci
def test_scalar_set_sample_splitting_tuple_rejected():
    """Reject tuple shorthand for scalar set_sample_splitting."""
    np.random.seed(3141)
    dml_data = make_plr_CCDDHNR2018(n_obs=10)
    dml_obj = PLR(dml_data)

    smpls = (np.arange(10), np.arange(10))
    msg = "all_smpls must be a list of folds; tuple shorthand is not supported"
    with pytest.raises(TypeError, match=msg):
        dml_obj.set_sample_splitting(smpls)
