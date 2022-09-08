import numpy as np
from statsmodels.tools.numdiff import approx_fprime, approx_hess3


def test_hfun(generate_copula_data):
    data = generate_copula_data['data']
    cop_obj = generate_copula_data['cop_obj']
    par = generate_copula_data['par']

    res = cop_obj.inv_hfun(par, cop_obj.hfun(par, data[:, 0], data[:, 1]), data[:, 1])

    assert np.allclose(data[:, 0],
                       res,
                       rtol=1e-9, atol=1e-4)


def test_hfun_numdiff(generate_copula_data):
    data = generate_copula_data['data']
    n_obs = data.shape[0]
    cop_obj = generate_copula_data['cop_obj']
    par = generate_copula_data['par']

    res = cop_obj.hfun(par, data[:, 0], data[:, 1])

    def cdf_for_numdiff(v, u):
        return cop_obj.cdf(par, u, v)

    res_num = np.full_like(res, np.nan)
    for i_obs in range(n_obs):
        res_num[i_obs] = approx_fprime(data[i_obs:i_obs+1, 1],
                                       cdf_for_numdiff,
                                       epsilon=1e-6,
                                       kwargs={'u': data[i_obs:i_obs+1, 0]},
                                       centered=True)

    assert np.allclose(res_num,
                       res,
                       rtol=1e-9, atol=1e-4)


def test_neg_ll_d_par_numdiff(generate_copula_data):
    data = generate_copula_data['data']
    cop_obj = generate_copula_data['cop_obj']
    par = generate_copula_data['par']

    res = cop_obj.neg_ll_d_par(par, data[:, 0], data[:, 1])

    res_num = approx_fprime(np.array([par]),
                            cop_obj.neg_ll,
                            epsilon=1e-6,
                            kwargs={'u': data[:, 0], 'v': data[:, 1]},
                            centered=True)

    assert np.allclose(res_num,
                       res,
                       rtol=1e-4, atol=1e-3)


def test_neg_ll_d_u_numdiff(generate_copula_data):
    data = generate_copula_data['data']
    n_obs = data.shape[0]
    cop_obj = generate_copula_data['cop_obj']
    par = generate_copula_data['par']

    res = cop_obj.ll_deriv(par, data[:, 0], data[:, 1], deriv='d_u')

    def ll_for_numdiff(u, v):
        return cop_obj.ll(par, u, v)

    res_num = np.full_like(res, np.nan)
    for i_obs in range(n_obs):
        res_num[i_obs] = approx_fprime(data[i_obs:i_obs+1, 0],
                                       ll_for_numdiff,
                                       epsilon=1e-6,
                                       kwargs={'v': data[i_obs:i_obs+1, 1]},
                                       centered=True)

    assert np.allclose(res_num,
                       res,
                       rtol=1e-4, atol=1e-3)


def test_neg_ll_d_v_numdiff(generate_copula_data):
    data = generate_copula_data['data']
    n_obs = data.shape[0]
    cop_obj = generate_copula_data['cop_obj']
    par = generate_copula_data['par']

    res = cop_obj.ll_deriv(par, data[:, 0], data[:, 1], deriv='d_v')

    def ll_for_numdiff(v, u):
        return cop_obj.ll(par, u, v)

    res_num = np.full_like(res, np.nan)
    for i_obs in range(n_obs):
        res_num[i_obs] = approx_fprime(data[i_obs:i_obs+1, 1],
                                       ll_for_numdiff,
                                       epsilon=1e-6,
                                       kwargs={'u': data[i_obs:i_obs+1, 0]},
                                       centered=True)

    assert np.allclose(res_num,
                       res,
                       rtol=1e-4, atol=1e-3)


def test_neg_ll_d2_u_v_par_numdiff(generate_copula_data):
    data = generate_copula_data['data']
    n_obs = data.shape[0]
    cop_obj = generate_copula_data['cop_obj']
    par = generate_copula_data['par']

    d2_ll_d_u_u = cop_obj.ll_deriv(par, data[:, 0], data[:, 1], deriv='d2_u_u')
    d2_ll_d_u_v = cop_obj.ll_deriv(par, data[:, 0], data[:, 1], deriv='d2_u_v')
    d2_ll_d_v_v = cop_obj.ll_deriv(par, data[:, 0], data[:, 1], deriv='d2_v_v')
    d2_ll_d_par_par = cop_obj.ll_deriv(par, data[:, 0], data[:, 1], deriv='d2_par_par')
    d2_ll_d_par_u = cop_obj.ll_deriv(par, data[:, 0], data[:, 1], deriv='d2_par_u')
    d2_ll_d_par_v = cop_obj.ll_deriv(par, data[:, 0], data[:, 1], deriv='d2_par_v')
    res = np.column_stack((d2_ll_d_par_par, d2_ll_d_par_u, d2_ll_d_par_v,
                           d2_ll_d_par_u, d2_ll_d_u_u, d2_ll_d_u_v,
                           d2_ll_d_par_v, d2_ll_d_u_v, d2_ll_d_v_v))

    def ll_for_numdiff(xx):
        return cop_obj.ll(xx[0:1], xx[1:2], xx[2:3])

    res_num = np.full_like(res, np.nan)
    for i_obs in range(n_obs):
        res_num[i_obs, :] = approx_hess3(np.append(par, data[i_obs:i_obs+1, :]),
                                         ll_for_numdiff,
                                         epsilon=1e-5).flatten()

    assert np.allclose(res_num,
                       res,
                       rtol=1e-3, atol=1e-2)
