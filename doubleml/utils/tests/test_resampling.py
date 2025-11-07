import pytest

from doubleml.utils.resampling import DoubleMLDoubleResampling


@pytest.mark.ci
def test_DoubleMLDoubleResampling_stratify():
    n_folds = 5
    n_folds_inner = 3
    n_rep = 2
    n_obs = 100
    stratify = [0] * 50 + [1] * 50

    obj_dml_double_resampling = DoubleMLDoubleResampling(
        n_folds=n_folds,
        n_folds_inner=n_folds_inner,
        n_rep=n_rep,
        n_obs=n_obs,
        stratify=stratify,
    )
    smpls, smpls_inner = obj_dml_double_resampling.split_samples()

    assert len(smpls) == n_rep
    assert len(smpls_inner) == n_rep

    for i_rep in range(n_rep):
        assert len(smpls[i_rep]) == n_folds
        assert len(smpls_inner[i_rep]) == n_folds

        for i_fold in range(n_folds):
            train_ind, _ = smpls[i_rep][i_fold]
            smpls_inner_rep_fold = smpls_inner[i_rep][i_fold]
            assert len(smpls_inner_rep_fold) == n_folds_inner

            for i_fold_inner in range(n_folds_inner):
                train_ind_inner, test_ind_inner = smpls_inner_rep_fold[i_fold_inner]
                assert set(train_ind_inner).issubset(set(train_ind))
                assert set(test_ind_inner).issubset(set(train_ind))


@pytest.mark.ci
def test_DoubleMLDoubleResampling_exceptions():
    msg = "n_folds must be greater than 1. You can use set_sample_splitting with a tuple to only use one fold."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLDoubleResampling(1, 5, 1, 100)

    msg = "n_folds_inner must be greater than 1. You can use set_sample_splitting with a tuple to only use one fold."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLDoubleResampling(5, 1, 1, 100)
