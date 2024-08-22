"""
Dummy test using fixed learner for sharp data
"""
import pytest
import numpy as np


@pytest.fixture(scope='module')
def data(rdd_fuzzy_right_data):
    return rdd_fuzzy_right_data


@pytest.fixture(scope='module',
                params=[-0.2, 0.0, 0.4])
def cutoff(request):
    return request.param


@pytest.fixture(scope='module',
                params=[0.05, 0.1])
def alpha(request):
    return request.param


@pytest.fixture(scope='module',
                params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope='module',
                params=[1, 2])
def p(request):
    return request.param


@pytest.mark.ci
def test_rdd_placebo_coef(predict_dummy, data, cutoff, alpha, p, n_rep):
    reference, actual = predict_dummy(
        data(cutoff=0.0),
        cutoff=cutoff,
        alpha=alpha,
        n_rep=n_rep,
        p=p
    )
    assert np.allclose(actual['coef'], reference['coef'], rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_rdd_nonplacebo_coef(predict_dummy, data, cutoff, alpha, p, n_rep):
    reference, actual = predict_dummy(
        data(cutoff=cutoff),
        cutoff=cutoff,
        alpha=alpha,
        n_rep=n_rep,
        p=p
    )
    assert np.allclose(actual['coef'], reference['coef'], rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_rdd_placebo_se(predict_dummy, data, cutoff, alpha, p, n_rep):
    reference, actual = predict_dummy(
        data(cutoff=0.0),
        cutoff=cutoff,
        alpha=alpha,
        n_rep=n_rep,
        p=p
    )
    assert np.allclose(actual['se'], reference['se'], rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_rdd_nonplacebo_se(predict_dummy, data, cutoff, alpha, p, n_rep):
    reference, actual = predict_dummy(
        data(cutoff=cutoff),
        cutoff=cutoff,
        alpha=alpha,
        n_rep=n_rep,
        p=p
    )
    assert np.allclose(actual['se'], reference['se'], rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_rdd_dmmy_placebo_ci(predict_dummy, data, cutoff, alpha, p, n_rep):
    reference, actual = predict_dummy(
        data(cutoff=0.0),
        cutoff=cutoff,
        alpha=alpha,
        n_rep=n_rep,
        p=p
    )
    assert np.allclose(actual['ci'], reference['ci'], rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_rdd_dmmy_nonplacebo_ci(predict_dummy, data, cutoff, alpha, p, n_rep):
    reference, actual = predict_dummy(
        data(cutoff=cutoff),
        cutoff=cutoff,
        alpha=alpha,
        n_rep=n_rep,
        p=p
    )
    assert np.allclose(actual['ci'], reference['ci'], rtol=1e-9, atol=1e-4)
