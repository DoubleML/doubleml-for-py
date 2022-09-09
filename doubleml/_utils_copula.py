import np as np
from abc import ABC, abstractmethod

from scipy.stats import kendalltau, norm, multivariate_normal
import scipy.integrate as integrate
from scipy.optimize import fmin_l_bfgs_b, root_scalar


class Copula(ABC):
    _par_bounds = None

    def __init__(self):
        return

    def mle_est(self, u, v):
        tau, _ = kendalltau(u, v)
        par_0 = self.tau2par(tau)
        par_hat, _, _ = fmin_l_bfgs_b(self.neg_ll,
                                      par_0,
                                      self.neg_ll_d_par,
                                      (u, v),
                                      bounds=self._par_bounds)
        return par_hat

    @staticmethod
    @abstractmethod
    def tau2par(tau):
        pass

    @staticmethod
    @abstractmethod
    def neg_ll(par, u, v):
        pass

    @staticmethod
    @abstractmethod
    def neg_ll_d_par(par, u, v):
        pass

    @staticmethod
    @abstractmethod
    def ll_deriv(par, u, v, deriv):
        pass

    @staticmethod
    @abstractmethod
    def inv_hfun(par, u, v):
        pass

    def sim(self, par, n_obs=100):
        u = np.random.uniform(size=(n_obs, 2))
        u[:, 0] = self.inv_hfun(par, u[:, 0], u[:, 1])
        return u


class ClaytonCopula(Copula):

    def __init__(self):
        super().__init__()
        self._par_bounds = [(0.0001, 28)]

    @staticmethod
    def tau2par(tau):
        return 2 * tau / (1 - tau)

    @staticmethod
    def cdf(par, u, v):
        res = (-1 + v**(-par) + u**(-par))**(-1/par)
        return res

    @staticmethod
    def pdf(par, u, v):
        # obtained with sympy
        res = u**par*v**par*(par + 1)*(-1 + v**(-par) + u**(-par))**(-1/par)/(u*v*(u**par*v**par - u**par - v**par)**2)
        return res

    @staticmethod
    def ll(par, u, v):
        # obtained with sympy
        res = np.log(u**par*v**par*(par + 1)*(-1 + v**(-par) + u**(-par))**(-1/par)/(u*v*(u**par*v**par - u**par - v**par)**2))
        return res

    @staticmethod
    def neg_ll(par, u, v):
        return -np.sum(ClaytonCopula.ll(par, u, v))

    @staticmethod
    def neg_ll_d_par(par, u, v):
        return -np.sum(ClaytonCopula.ll_deriv(par, u, v, 'd_par'))

    @staticmethod
    def ll_deriv(par, u, v, deriv):

        # u[u < trimming_threshold] = trimming_threshold
        # u[u > 1 - trimming_threshold] = 1 - trimming_threshold
        # v[v < trimming_threshold] = trimming_threshold
        # v[v > 1 - trimming_threshold] = 1 - trimming_threshold

        # TODO: Add checks for the parameter
        if deriv == 'd_par':
            # obtained with sympy
            res = (-par**3*u**par*v**par*np.log(u) - par**3*u**par*v**par*np.log(v) + par**3*u**par*np.log(u) - par**3*u**par*np.log(v) - par**3*v**par*np.log(u) + par**3*v**par*np.log(v) - par**2*u**par*v**par*np.log(u) - par**2*u**par*v**par*np.log(v) + par**2*u**par*v**par + par**2*u**par*np.log(u) - 2*par**2*u**par*np.log(v) - par**2*u**par - 2*par**2*v**par*np.log(u) + par**2*v**par*np.log(v) - par**2*v**par + par*u**par*v**par*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)) - par*u**par*np.log(v) - par*u**par*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)) - par*v**par*np.log(u) - par*v**par*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)) + u**par*v**par*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)) - u**par*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)) - v**par*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)))/(par**2*(par + 1)*(u**par*v**par - u**par - v**par))
        elif deriv == 'd_u':
            # obtained with sympy
            res = -(par*u**par*v**par - par*u**par + par*v**par + u**par*v**par - u**par)/(u*(u**par*v**par - u**par - v**par))
        elif deriv == 'd_v':
            res = ClaytonCopula.ll_deriv(par, v, u, deriv='d_u')
        elif deriv == 'd2_u_u':
            # obtained with sympy
            res = (2*par**2*u**par*v**(2*par) - 2*par**2*u**par*v**par + par*u**(2*par)*v**(2*par) - 2*par*u**(2*par)*v**par + par*u**(2*par) + par*u**par*v**(2*par) - par*u**par*v**par - par*v**(2*par) + u**(2*par)*v**(2*par) - 2*u**(2*par)*v**par + u**(2*par) - u**par*v**(2*par) + u**par*v**par)/(u**2*(u**par*v**par - u**par - v**par)**2)
        elif deriv == 'd2_v_v':
            res = ClaytonCopula.ll_deriv(par, v, u, deriv='d2_u_u')
        elif deriv == 'd2_par_par':
            # obtained with sympy
            res = -(-2*par**5*u**(2*par)*v**par*np.log(v)**2 - 2*par**5*u**par*v**(2*par)*np.log(u)**2 + 2*par**5*u**par*v**par*np.log(u)**2 - 4*par**5*u**par*v**par*np.log(u)*np.log(v) + 2*par**5*u**par*v**par*np.log(v)**2 - 5*par**4*u**(2*par)*v**par*np.log(v)**2 - 5*par**4*u**par*v**(2*par)*np.log(u)**2 + 5*par**4*u**par*v**par*np.log(u)**2 - 10*par**4*u**par*v**par*np.log(u)*np.log(v) + 5*par**4*u**par*v**par*np.log(v)**2 + par**3*u**(2*par)*v**(2*par) - 4*par**3*u**(2*par)*v**par*np.log(v)**2 - 2*par**3*u**(2*par)*v**par*np.log(v) - 2*par**3*u**(2*par)*v**par + 2*par**3*u**(2*par)*np.log(v) + par**3*u**(2*par) - 4*par**3*u**par*v**(2*par)*np.log(u)**2 - 2*par**3*u**par*v**(2*par)*np.log(u) - 2*par**3*u**par*v**(2*par) + 4*par**3*u**par*v**par*np.log(u)**2 - 8*par**3*u**par*v**par*np.log(u)*np.log(v) + 2*par**3*u**par*v**par*np.log(u) + 4*par**3*u**par*v**par*np.log(v)**2 + 2*par**3*u**par*v**par*np.log(v) + 2*par**3*u**par*v**par + 2*par**3*v**(2*par)*np.log(u) + par**3*v**(2*par) + 2*par**2*u**(2*par)*v**(2*par)*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)) - par**2*u**(2*par)*v**par*np.log(v)**2 - 4*par**2*u**(2*par)*v**par*np.log(v) - 4*par**2*u**(2*par)*v**par*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)) + 4*par**2*u**(2*par)*np.log(v) + 2*par**2*u**(2*par)*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)) - par**2*u**par*v**(2*par)*np.log(u)**2 - 4*par**2*u**par*v**(2*par)*np.log(u) - 4*par**2*u**par*v**(2*par)*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)) + par**2*u**par*v**par*np.log(u)**2 - 2*par**2*u**par*v**par*np.log(u)*np.log(v) + 4*par**2*u**par*v**par*np.log(u) + par**2*u**par*v**par*np.log(v)**2 + 4*par**2*u**par*v**par*np.log(v) + 4*par**2*u**par*v**par*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)) + 4*par**2*v**(2*par)*np.log(u) + 2*par**2*v**(2*par)*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)) + 4*par*u**(2*par)*v**(2*par)*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)) - 2*par*u**(2*par)*v**par*np.log(v) - 8*par*u**(2*par)*v**par*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)) + 2*par*u**(2*par)*np.log(v) + 4*par*u**(2*par)*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)) - 2*par*u**par*v**(2*par)*np.log(u) - 8*par*u**par*v**(2*par)*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)) + 2*par*u**par*v**par*np.log(u) + 2*par*u**par*v**par*np.log(v) + 8*par*u**par*v**par*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)) + 2*par*v**(2*par)*np.log(u) + 4*par*v**(2*par)*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)) + 2*u**(2*par)*v**(2*par)*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)) - 4*u**(2*par)*v**par*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)) + 2*u**(2*par)*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)) - 4*u**par*v**(2*par)*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)) + 4*u**par*v**par*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)) + 2*v**(2*par)*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)))/(par**3*(par + 1)**2*(u**par*v**par - u**par - v**par)**2)
        elif deriv == 'd2_u_v':
            # obtained with sympy
            res = par*u**par*v**par*(2*par + 1)/(u*v*(u**par*v**par - u**par - v**par)**2)
        elif deriv == 'd2_par_u':
            # obtained with sympy
            res = (2*par*u**par*v**(2*par)*np.log(u) - 2*par*u**par*v**par*np.log(u) + 2*par*u**par*v**par*np.log(v) - u**(2*par)*v**(2*par) + 2*u**(2*par)*v**par - u**(2*par) + u**par*v**(2*par)*np.log(u) - u**par*v**par*np.log(u) + u**par*v**par*np.log(v) + v**(2*par))/(u*(u**par*v**par - u**par - v**par)**2)
        else:
            assert deriv == 'd2_par_v'
            res = ClaytonCopula.ll_deriv(par, v, u, deriv='d2_par_u')
        return res

    @staticmethod
    def hfun(par, u, v):
        # TODO: Add checks for the parameter
        # obtained with sympy
        res = v**(-par)*(-1 + v**(-par) + u**(-par))**(-1/par)/(v*(-1 + v**(-par) + u**(-par)))
        return res

    @staticmethod
    def inv_hfun(par, u, v):
        # TODO: Add checks for the parameter
        h1 = -1/par
        h2 = -par/(1+par)
        res = np.power(np.power(v, -par) * (np.power(u, h2)-1) + 1, h1)
        return res


class FrankCopula(Copula):

    def __init__(self):
        super().__init__()
        self._par_bounds = [(-40, 40)]

    @staticmethod
    def par2tau(par):
        # ToDO: Check and compare with R
        debye_fun = integrate.quad(lambda x: x / np.expm1(x), 0, par)[0]
        tau = 1 - 4/par*(1-debye_fun/par)
        return tau

    @staticmethod
    def tau2par(tau):
        # ToDO: Check and compare with R
        tau_l = FrankCopula().par2tau(-40)
        tau_u = FrankCopula().par2tau(40)
        if (tau < tau_l) | (tau > tau_u):
            raise ValueError(f'Choose Kendall tau between {tau_l} and {tau_u}.')
        if tau == 0.:
            par = 0.
        else:
            if tau > 0:
                bracket = [0.0001, 40]
            else:
                bracket = [-40, -0.0001]
            root_res = root_scalar(lambda xx: FrankCopula().par2tau(xx) - tau,
                                   bracket=bracket,
                                   method='brentq')
            par = root_res.root
        return par

    @staticmethod
    def cdf(par, u, v):
        return -np.log((np.expm1(-par) + np.expm1(-par*u)*np.expm1(-par*v))/np.expm1(-par))/par

    @staticmethod
    def pdf(par, u, v):
        # obtained with sympy
        res = par * np.exp(-par * u) * np.exp(-par * v) / (
                    -np.expm1(-par) - np.expm1(-par * u) * np.expm1(-par * v)) + par * np.exp(-par * u) * np.exp(
            -par * v) * np.expm1(-par * u) * np.expm1(-par * v) / (
                    -np.expm1(-par) - np.expm1(-par * u) * np.expm1(-par * v)) ** 2
        return res

    @staticmethod
    def ll(par, u, v):
        # obtained with sympy
        res = np.log(-par*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par)
                     / (np.expm1(-par) + np.expm1(-par*u)*np.expm1(-par*v))**2)
        return res

    @staticmethod
    def neg_ll(par, u, v):
        return -np.sum(FrankCopula.ll(par, u, v))

    @staticmethod
    def neg_ll_d_par(par, u, v):
        return -np.sum(FrankCopula.ll_deriv(par, u, v, 'd_par'))

    @staticmethod
    def ll_deriv(par, u, v, deriv):

        # u[u < trimming_threshold] = trimming_threshold
        # u[u > 1 - trimming_threshold] = 1 - trimming_threshold
        # v[v < trimming_threshold] = trimming_threshold
        # v[v > 1 - trimming_threshold] = 1 - trimming_threshold

        # TODO: Add checks for the parameter
        if deriv == 'd_par':
            # obtained with sympy

            res = (-par*u*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) - par*u*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 - par*u*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 - par*v*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) - par*v*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 - par*v*np.exp(-par*u)*np.exp(-2*par*v)*np.expm1(-par*u)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + par*(-u*np.exp(-par*u)*np.expm1(-par*v) - v*np.exp(-par*v)*np.expm1(-par*u) - np.exp(-par))*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + par*(-2*u*np.exp(-par*u)*np.expm1(-par*v) - 2*v*np.exp(-par*v)*np.expm1(-par*u) - 2*np.exp(-par))*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3 + np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) + np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2)/(par*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) + par*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2)
        elif deriv == 'd_u':
            # obtained with sympy
            res = (-par**2*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) - par**2*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 - 2*par**2*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 - 2*par**2*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)**2/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3)/(par*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) + par*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2)
        elif deriv == 'd_v':
            res = FrankCopula.ll_deriv(par, v, u, deriv='d_u')
        elif deriv == 'd2_u_u':
            # obtained with sympy
            res = (par**3*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) + par**3*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + 6*par**3*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + 6*par**3*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)**2/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3 + 6*par**3*np.exp(-3*par*u)*np.exp(-par*v)*np.expm1(-par*v)**2/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3 + 6*par**3*np.exp(-3*par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)**3/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**4)/(par*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) + par*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2) + (-par**2*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) - par**2*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 - 2*par**2*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 - 2*par**2*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)**2/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3)*(par**2*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) + par**2*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + 2*par**2*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + 2*par**2*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)**2/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3)/(par*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) + par*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2)**2
        elif deriv == 'd2_v_v':
            res = FrankCopula.ll_deriv(par, v, u, deriv='d2_u_u')
        elif deriv == 'd2_par_par':
            # obtained with sympy
            res = (par*u**2*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) + par*u**2*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + 3*par*u**2*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + 2*par*u*v*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) + 2*par*u*v*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + 2*par*u*v*np.exp(-par*u)*np.exp(-2*par*v)*np.expm1(-par*u)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + 2*par*u*v*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + 2*par*u*v*np.exp(-2*par*u)*np.exp(-2*par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 - 2*par*u*(-u*np.exp(-par*u)*np.expm1(-par*v) - v*np.exp(-par*v)*np.expm1(-par*u) - np.exp(-par))*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 - 2*par*u*(-2*u*np.exp(-par*u)*np.expm1(-par*v) - 2*v*np.exp(-par*v)*np.expm1(-par*u) - 2*np.exp(-par))*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3 - 2*par*u*(-2*u*np.exp(-par*u)*np.expm1(-par*v) - 2*v*np.exp(-par*v)*np.expm1(-par*u) - 2*np.exp(-par))*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3 + par*v**2*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) + par*v**2*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + 3*par*v**2*np.exp(-par*u)*np.exp(-2*par*v)*np.expm1(-par*u)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 - 2*par*v*(-u*np.exp(-par*u)*np.expm1(-par*v) - v*np.exp(-par*v)*np.expm1(-par*u) - np.exp(-par))*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 - 2*par*v*(-2*u*np.exp(-par*u)*np.expm1(-par*v) - 2*v*np.exp(-par*v)*np.expm1(-par*u) - 2*np.exp(-par))*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3 - 2*par*v*(-2*u*np.exp(-par*u)*np.expm1(-par*v) - 2*v*np.exp(-par*v)*np.expm1(-par*u) - 2*np.exp(-par))*np.exp(-par*u)*np.exp(-2*par*v)*np.expm1(-par*u)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3 + par*(u**2*np.exp(-par*u)*np.expm1(-par*v) + 2*u*v*np.exp(-par*u)*np.exp(-par*v) + v**2*np.exp(-par*v)*np.expm1(-par*u) + np.exp(-par))*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + par*(-2*u*np.exp(-par*u)*np.expm1(-par*v) - 2*v*np.exp(-par*v)*np.expm1(-par*u) - 2*np.exp(-par))*(-u*np.exp(-par*u)*np.expm1(-par*v) - v*np.exp(-par*v)*np.expm1(-par*u) - np.exp(-par))*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3 + par*(2*u**2*np.exp(-par*u)*np.expm1(-par*v) + 4*u*v*np.exp(-par*u)*np.exp(-par*v) + 2*v**2*np.exp(-par*v)*np.expm1(-par*u) + 2*np.exp(-par))*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3 + par*(-3*u*np.exp(-par*u)*np.expm1(-par*v) - 3*v*np.exp(-par*v)*np.expm1(-par*u) - 3*np.exp(-par))*(-2*u*np.exp(-par*u)*np.expm1(-par*v) - 2*v*np.exp(-par*v)*np.expm1(-par*u) - 2*np.exp(-par))*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**4 - 2*u*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) - 2*u*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 - 2*u*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 - 2*v*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) - 2*v*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 - 2*v*np.exp(-par*u)*np.exp(-2*par*v)*np.expm1(-par*u)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + 2*(-u*np.exp(-par*u)*np.expm1(-par*v) - v*np.exp(-par*v)*np.expm1(-par*u) - np.exp(-par))*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + 2*(-2*u*np.exp(-par*u)*np.expm1(-par*v) - 2*v*np.exp(-par*v)*np.expm1(-par*u) - 2*np.exp(-par))*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3)/(par*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) + par*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2) + (-par*u*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) - par*u*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 - par*u*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 - par*v*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) - par*v*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 - par*v*np.exp(-par*u)*np.exp(-2*par*v)*np.expm1(-par*u)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + par*(-u*np.exp(-par*u)*np.expm1(-par*v) - v*np.exp(-par*v)*np.expm1(-par*u) - np.exp(-par))*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + par*(-2*u*np.exp(-par*u)*np.expm1(-par*v) - 2*v*np.exp(-par*v)*np.expm1(-par*u) - 2*np.exp(-par))*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3 + np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) + np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2)*(par*u*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) + par*u*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + par*u*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + par*v*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) + par*v*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + par*v*np.exp(-par*u)*np.exp(-2*par*v)*np.expm1(-par*u)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 - par*(-u*np.exp(-par*u)*np.expm1(-par*v) - v*np.exp(-par*v)*np.expm1(-par*u) - np.exp(-par))*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 - par*(-2*u*np.exp(-par*u)*np.expm1(-par*v) - 2*v*np.exp(-par*v)*np.expm1(-par*u) - 2*np.exp(-par))*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3 - np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) - np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2)/(par*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) + par*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2)**2
        elif deriv == 'd2_u_v':
            # obtained with sympy
            res = (par**3*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) + par**3*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + 2*par**3*np.exp(-par*u)*np.exp(-2*par*v)*np.expm1(-par*u)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + 2*par**3*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + 2*par**3*np.exp(-2*par*u)*np.exp(-2*par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + 2*par**3*np.exp(-par*u)*np.exp(-2*par*v)*np.expm1(-par*u)**2*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3 + 2*par**3*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)**2/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3 + 8*par**3*np.exp(-2*par*u)*np.exp(-2*par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3 + 6*par**3*np.exp(-2*par*u)*np.exp(-2*par*v)*np.expm1(-par*u)**2*np.expm1(-par*v)**2/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**4)/(par*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) + par*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2) + (-par**2*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) - par**2*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 - 2*par**2*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 - 2*par**2*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)**2/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3)*(par**2*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) + par**2*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + 2*par**2*np.exp(-par*u)*np.exp(-2*par*v)*np.expm1(-par*u)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + 2*par**2*np.exp(-par*u)*np.exp(-2*par*v)*np.expm1(-par*u)**2*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3)/(par*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) + par*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2)**2
        elif deriv == 'd2_par_u':
            # obtained with sympy
            res = (par**2*u*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) + par**2*u*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + 4*par**2*u*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + 2*par**2*u*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)**2/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3 + 2*par**2*u*np.exp(-3*par*u)*np.exp(-par*v)*np.expm1(-par*v)**2/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3 + par**2*v*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) + par**2*v*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + par**2*v*np.exp(-par*u)*np.exp(-2*par*v)*np.expm1(-par*u)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + 2*par**2*v*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + par**2*v*np.exp(-2*par*u)*np.exp(-2*par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + 2*par**2*v*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)**2/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3 + 2*par**2*v*np.exp(-2*par*u)*np.exp(-2*par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3 - par**2*(-u*np.exp(-par*u)*np.expm1(-par*v) - v*np.exp(-par*v)*np.expm1(-par*u) - np.exp(-par))*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 - par**2*(-2*u*np.exp(-par*u)*np.expm1(-par*v) - 2*v*np.exp(-par*v)*np.expm1(-par*u) - 2*np.exp(-par))*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3 - par**2*(-2*u*np.exp(-par*u)*np.expm1(-par*v) - 2*v*np.exp(-par*v)*np.expm1(-par*u) - 2*np.exp(-par))*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3 - 2*par**2*(-u*np.exp(-par*u)*np.expm1(-par*v) - v*np.exp(-par*v)*np.expm1(-par*u) - np.exp(-par))*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3 - 3*par**2*(-2*u*np.exp(-par*u)*np.expm1(-par*v) - 2*v*np.exp(-par*v)*np.expm1(-par*u) - 2*np.exp(-par))*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)**2/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**4 - 2*par*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) + par*(par*u*np.exp(-par*u)*np.expm1(-par*v) + par*v*np.exp(-par*u)*np.exp(-par*v) - np.exp(-par*u)*np.expm1(-par*v))*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 - 2*par*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 - 3*par*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + par*(2*par*u*np.exp(-par*u)*np.expm1(-par*v) + 2*par*v*np.exp(-par*u)*np.exp(-par*v) - 2*np.exp(-par*u)*np.expm1(-par*v))*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3 - 2*par*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)**2/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3)/(par*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) + par*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2) + (par**2*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) + par**2*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + 2*par**2*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + 2*par**2*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)**2/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3)*(-par*u*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) - par*u*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 - par*u*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 - par*v*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) - par*v*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 - par*v*np.exp(-par*u)*np.exp(-2*par*v)*np.expm1(-par*u)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + par*(-u*np.exp(-par*u)*np.expm1(-par*v) - v*np.exp(-par*v)*np.expm1(-par*u) - np.exp(-par))*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2 + par*(-2*u*np.exp(-par*u)*np.expm1(-par*v) - 2*v*np.exp(-par*v)*np.expm1(-par*u) - 2*np.exp(-par))*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3 + np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) + np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2)/(par*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)) + par*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2)**2
        else:
            assert deriv == 'd2_par_v'
            res = FrankCopula.ll_deriv(par, v, u, deriv='d2_par_u')
        return res

    @staticmethod
    def hfun(par, u, v):
        # TODO: Add checks for the parameter
        # obtained with sympy
        res = -np.exp(-par*v)*np.expm1(-par*u)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))
        return res

    @staticmethod
    def inv_hfun(par, u, v):
        # TODO: Add checks for the parameter
        res = -1/par * np.log(1 + np.expm1(-par)/(np.exp(-par*v) * (1/u-1) + 1))
        return res


class GaussianCopula(Copula):

    def __init__(self):
        super().__init__()
        self._par_bounds = [(-0.999, 0.999)]

    @staticmethod
    def tau2par(tau):
        return np.sin(np.pi * tau / 2)

    @staticmethod
    def cdf(par, u, v):
        x = norm.ppf(u)
        y = norm.ppf(v)
        return multivariate_normal.cdf(np.column_stack((x,y)),
                                       mean=[0., 0.],
                                       cov=[[1., par], [par, 1.]])

    @staticmethod
    def pdf(par, u, v):
        x = norm.ppf(u)
        y = norm.ppf(v)
        # obtained with sympy
        res = np.exp((-par**2*(x**2 + y**2) + 2*par*x*y)/(2 - 2*par**2))/np.sqrt(1 - par**2)
        return res

    @staticmethod
    def ll(par, u, v):
        x = norm.ppf(u)
        y = norm.ppf(v)
        # obtained with sympy
        res = -1/2*np.log(-(par - 1)*(par + 1)) - (par**2*(x**2 + y**2) - 2*par*x*y)/(2 - 2*par**2)
        return res

    @staticmethod
    def neg_ll(par, u, v):
        return -np.sum(GaussianCopula.ll(par, u, v))

    @staticmethod
    def neg_ll_d_par(par, u, v):
        return -np.sum(GaussianCopula.ll_deriv(par, u, v, 'd_par'))

    @staticmethod
    def _d_ll_d_x(par, x, y):
        res = par*(par*x - y)/((par - 1)*(par + 1))
        return res

    @staticmethod
    def ll_deriv(par, u, v, deriv):
        # TODO: Add checks for the parameter
        x = norm.ppf(u)
        y = norm.ppf(v)
        if deriv == 'd_par':
            # obtained with sympy
            res = -(par**3 - par**2*x*y + par*x**2 + par*y**2 - par - x*y)/((par - 1)**2*(par + 1)**2)
        elif deriv == 'd_u':
            x_deriv_u = np.sqrt(2 * np.pi) * np.exp(x ** 2 / 2)
            # obtained with sympy
            res = x_deriv_u * GaussianCopula._d_ll_d_x(par, x, y)
        elif deriv == 'd_v':
            res = GaussianCopula.ll_deriv(par, v, u, deriv='d_u')
        elif deriv == 'd2_u_u':
            x_deriv_u = np.sqrt(2 * np.pi) * np.exp(x ** 2 / 2)
            x_deriv2_u_u = 2 * np.pi * x * np.exp(x ** 2)
            ll_deriv_x = GaussianCopula._d_ll_d_x(par, x, y)
            ll_deriv2_x_x = par**2/((par - 1)*(par + 1))
            # obtained with sympy
            res = x_deriv2_u_u * ll_deriv_x + x_deriv_u ** 2 * ll_deriv2_x_x
        elif deriv == 'd2_v_v':
            res = GaussianCopula.ll_deriv(par, v, u, deriv='d2_u_u')
        elif deriv == 'd2_par_par':
            # obtained with sympy
            res = (par**4 - 2*par**3*x*y + 3*par**2*x**2 + 3*par**2*y**2 - 6*par*x*y + x**2 + y**2 - 1)/((par - 1)**3*(par + 1)**3)
        elif deriv == 'd2_u_v':
            x_deriv_u = np.sqrt(2 * np.pi) * np.exp(x ** 2 / 2)
            y_deriv_v = np.sqrt(2 * np.pi) * np.exp(y ** 2 / 2)
            # obtained with sympy
            ll_deriv2_x_y = 2*par/(2 - 2*par**2)
            res = x_deriv_u * y_deriv_v * ll_deriv2_x_y
        elif deriv == 'd2_par_u':
            x_deriv_u = np.sqrt(2 * np.pi) * np.exp(x ** 2 / 2)
            # obtained with sympy
            ll_deriv2_par_x = -(-par**2*y + 2*par*x - y)/((par - 1)**2*(par + 1)**2)
            res = x_deriv_u * ll_deriv2_par_x
        else:
            assert deriv == 'd2_par_v'
            res = GaussianCopula.ll_deriv(par, v, u, deriv='d2_par_u')

        return res

    @staticmethod
    def hfun(par, u, v):
        # TODO: Add checks for the parameter
        x = norm.ppf(u)
        y = norm.ppf(v)
        res = norm.cdf((x - par * y) / np.sqrt(1 - par ** 2))
        return res

    @staticmethod
    def inv_hfun(par, u, v):
        # TODO: Add checks for the parameter
        x = norm.ppf(u)
        y = norm.ppf(v)
        res = norm.cdf(x * np.sqrt(1 - par ** 2) + par * y)
        return res


class GumbelCopula(Copula):

    def __init__(self):
        super().__init__()
        self._par_bounds = [(1.0, 20)]

    @staticmethod
    def tau2par(tau):
        return 1/(1 - tau)

    @staticmethod
    def cdf(par, u, v):
        return np.exp(-((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)))

    @staticmethod
    def pdf(par, u, v):
        # obtained with sympy
        res = (-np.log(u)) ** par * (-np.log(v)) ** par * ((-np.log(u)) ** par + (-np.log(v)) ** par) ** (
                    par ** (-1.0)) * (
                      par + ((-np.log(u)) ** par + (-np.log(v)) ** par) ** (par ** (-1.0)) - 1) * np.exp(
            -((-np.log(u)) ** par + (-np.log(v)) ** par) ** (par ** (-1.0))) / (
                      u * v * ((-np.log(u)) ** par + (-np.log(v)) ** par) ** 2 * np.log(u) * np.log(v))
        return res

    @staticmethod
    def ll(par, u, v):
        return np.log((-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*(par + ((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) - 1)*np.exp(-((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)))/(u*v*((-np.log(u))**par + (-np.log(v))**par)**2*np.log(u)*np.log(v)))

    @staticmethod
    def neg_ll(par, u, v):
        return -np.sum(GumbelCopula.ll(par, u, v))

    @staticmethod
    def neg_ll_d_par(par, u, v):
        return -np.sum(GumbelCopula.ll_deriv(par, u, v, 'd_par'))

    @staticmethod
    def ll_deriv(par, u, v, deriv):

        # u[u < trimming_threshold] = trimming_threshold
        # u[u > 1 - trimming_threshold] = 1 - trimming_threshold
        # v[v < trimming_threshold] = trimming_threshold
        # v[v > 1 - trimming_threshold] = 1 - trimming_threshold

        # TODO: Add checks for the parameter
        if deriv == 'd_par':
            res = (-par**3*(-np.log(u))**par*np.log(-np.log(u)) + par**3*(-np.log(u))**par*np.log(-np.log(v)) + par**3*(-np.log(v))**par*np.log(-np.log(u)) - par**3*(-np.log(v))**par*np.log(-np.log(v)) - 2*par**2*(-np.log(u))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u)) + par**2*(-np.log(u))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(v)) + 2*par**2*(-np.log(u))**par*np.log(-np.log(u)) - par**2*(-np.log(u))**par*np.log(-np.log(v)) + par**2*(-np.log(u))**par + par**2*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u)) - 2*par**2*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(v)) - par**2*(-np.log(v))**par*np.log(-np.log(u)) + 2*par**2*(-np.log(v))**par*np.log(-np.log(v)) + par**2*(-np.log(v))**par - par*(-np.log(u))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(u)) + par*(-np.log(u))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par) + 3*par*(-np.log(u))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u)) - par*(-np.log(u))**par*np.log((-np.log(u))**par + (-np.log(v))**par) - par*(-np.log(u))**par*np.log(-np.log(u)) - par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(v)) + par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par) + 3*par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(v)) - par*(-np.log(v))**par*np.log((-np.log(u))**par + (-np.log(v))**par) - par*(-np.log(v))**par*np.log(-np.log(v)) + (-np.log(u))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par) - 3*(-np.log(u))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par) + (-np.log(u))**par*np.log((-np.log(u))**par + (-np.log(v))**par) + (-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par) - 3*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par) + (-np.log(v))**par*np.log((-np.log(u))**par + (-np.log(v))**par))/(par**2*((-np.log(u))**par + (-np.log(v))**par)*(par + ((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) - 1))
        elif deriv == 'd_u':
            res = -(par**2*(-np.log(u))**par - par**2*(-np.log(v))**par + 2*par*(-np.log(u))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) + par*(-np.log(u))**par*np.log(u) - par*(-np.log(u))**par - par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) + par*(-np.log(v))**par*np.log(u) + 2*par*(-np.log(v))**par + (-np.log(u))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par) + (-np.log(u))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(u) - 2*(-np.log(u))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) - (-np.log(u))**par*np.log(u) + (-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(u) + (-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) - (-np.log(v))**par*np.log(u) - (-np.log(v))**par)/(u*((-np.log(u))**par + (-np.log(v))**par)*(par + ((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) - 1)*np.log(u))
        elif deriv == 'd_v':
            res = GumbelCopula.ll_deriv(par, v, u, deriv='d_u')
        elif deriv == 'd2_u_u':
            res = (-2*par**4*(-np.log(u))**par*(-np.log(v))**par + par**3*(-np.log(u))**(2*par)*np.log(u) + par**3*(-np.log(u))**(2*par) - 5*par**3*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) + 5*par**3*(-np.log(u))**par*(-np.log(v))**par - par**3*(-np.log(v))**(2*par)*np.log(u) - par**3*(-np.log(v))**(2*par) + 3*par**2*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(u) + 2*par**2*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) + par**2*(-np.log(u))**(2*par)*np.log(u)**2 - 2*par**2*(-np.log(u))**(2*par)*np.log(u) - 2*par**2*(-np.log(u))**(2*par) - 4*par**2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par) + par**2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(u) + 10*par**2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) + 2*par**2*(-np.log(u))**par*(-np.log(v))**par*np.log(u)**2 + par**2*(-np.log(u))**par*(-np.log(v))**par*np.log(u) - 3*par**2*(-np.log(u))**par*(-np.log(v))**par - 2*par**2*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(u) - 2*par**2*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) + par**2*(-np.log(v))**(2*par)*np.log(u)**2 + 3*par**2*(-np.log(v))**(2*par)*np.log(u) + 3*par**2*(-np.log(v))**(2*par) + 3*par*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(u) + par*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par) + 2*par*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(u)**2 - 5*par*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(u) - 2*par*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) - 2*par*(-np.log(u))**(2*par)*np.log(u)**2 + par*(-np.log(u))**(2*par)*np.log(u) + par*(-np.log(u))**(2*par) - par*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(3/par) + 2*par*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(u) + 6*par*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par) + 4*par*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(u)**2 - par*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(u) - 5*par*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) - 4*par*(-np.log(u))**par*(-np.log(v))**par*np.log(u)**2 - 2*par*(-np.log(u))**par*(-np.log(v))**par*np.log(u) - par*(-np.log(u))**par*(-np.log(v))**par - par*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(u) - par*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par) + 2*par*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(u)**2 + 4*par*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(u) + 4*par*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) - 2*par*(-np.log(v))**(2*par)*np.log(u)**2 - 3*par*(-np.log(v))**(2*par)*np.log(u) - 3*par*(-np.log(v))**(2*par) + (-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log(u) + (-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(u)**2 - 3*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(u) - (-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par) - 2*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(u)**2 + 2*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(u) + (-np.log(u))**(2*par)*np.log(u)**2 + (-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log(u) + (-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(3/par) + 2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(u)**2 - 2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(u) - 2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par) - 4*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(u)**2 + 2*(-np.log(u))**par*(-np.log(v))**par*np.log(u)**2 + (-np.log(u))**par*(-np.log(v))**par*np.log(u) + (-np.log(u))**par*(-np.log(v))**par + (-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(u)**2 + (-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(u) + (-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par) - 2*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(u)**2 - 2*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(u) - 2*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) + (-np.log(v))**(2*par)*np.log(u)**2 + (-np.log(v))**(2*par)*np.log(u) + (-np.log(v))**(2*par))/(u**2*((-np.log(u))**par + (-np.log(v))**par)**2*(par + ((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) - 1)**2*np.log(u)**2)
        elif deriv == 'd2_v_v':
            res = GumbelCopula.ll_deriv(par, v, u, deriv='d2_u_u')
        elif deriv == 'd2_par_par':
            res = -(2*par**6*(-np.log(u))**par*(-np.log(v))**par*np.log(-np.log(u))**2 - 4*par**6*(-np.log(u))**par*(-np.log(v))**par*np.log(-np.log(u))*np.log(-np.log(v)) + 2*par**6*(-np.log(u))**par*(-np.log(v))**par*np.log(-np.log(v))**2 + 5*par**5*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u))**2 - 10*par**5*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u))*np.log(-np.log(v)) + 5*par**5*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(v))**2 - 5*par**5*(-np.log(u))**par*(-np.log(v))**par*np.log(-np.log(u))**2 + 10*par**5*(-np.log(u))**par*(-np.log(v))**par*np.log(-np.log(u))*np.log(-np.log(v)) - 5*par**5*(-np.log(u))**par*(-np.log(v))**par*np.log(-np.log(v))**2 + par**4*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u))**2 - 2*par**4*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u)) + 2*par**4*(-np.log(u))**(2*par)*np.log(-np.log(u)) + par**4*(-np.log(u))**(2*par) + 4*par**4*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(u))**2 - 8*par**4*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(u))*np.log(-np.log(v)) + 4*par**4*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(v))**2 - 9*par**4*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u))**2 + 20*par**4*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u))*np.log(-np.log(v)) - 2*par**4*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u)) - 9*par**4*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(v))**2 - 2*par**4*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(v)) + 4*par**4*(-np.log(u))**par*(-np.log(v))**par*np.log(-np.log(u))**2 - 8*par**4*(-np.log(u))**par*(-np.log(v))**par*np.log(-np.log(u))*np.log(-np.log(v)) + 2*par**4*(-np.log(u))**par*(-np.log(v))**par*np.log(-np.log(u)) + 4*par**4*(-np.log(u))**par*(-np.log(v))**par*np.log(-np.log(v))**2 + 2*par**4*(-np.log(u))**par*(-np.log(v))**par*np.log(-np.log(v)) + 2*par**4*(-np.log(u))**par*(-np.log(v))**par + par**4*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(v))**2 - 2*par**4*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(v)) + 2*par**4*(-np.log(v))**(2*par)*np.log(-np.log(v)) + par**4*(-np.log(v))**(2*par) + 2*par**3*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(u))**2 - 4*par**3*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(u)) - 2*par**3*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par)*np.log(-np.log(u)) + 2*par**3*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par) - 3*par**3*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u))**2 + 12*par**3*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u)) - 2*par**3*(-np.log(u))**(2*par)*np.log((-np.log(u))**par + (-np.log(v))**par) - 4*par**3*(-np.log(u))**(2*par)*np.log(-np.log(u)) + par**3*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log(-np.log(u))**2 - 2*par**3*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log(-np.log(u))*np.log(-np.log(v)) + par**3*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log(-np.log(v))**2 - 4*par**3*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(u))**2 + 12*par**3*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(u))*np.log(-np.log(v)) - 4*par**3*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(u)) - 4*par**3*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(v))**2 - 4*par**3*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(v)) - 2*par**3*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par)*np.log(-np.log(u)) - 2*par**3*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par)*np.log(-np.log(v)) + 4*par**3*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par) + 4*par**3*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u))**2 - 14*par**3*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u))*np.log(-np.log(v)) + 12*par**3*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u)) + 4*par**3*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(v))**2 + 12*par**3*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(v)) - 4*par**3*(-np.log(u))**par*(-np.log(v))**par*np.log((-np.log(u))**par + (-np.log(v))**par) - par**3*(-np.log(u))**par*(-np.log(v))**par*np.log(-np.log(u))**2 + 2*par**3*(-np.log(u))**par*(-np.log(v))**par*np.log(-np.log(u))*np.log(-np.log(v)) - 4*par**3*(-np.log(u))**par*(-np.log(v))**par*np.log(-np.log(u)) - par**3*(-np.log(u))**par*(-np.log(v))**par*np.log(-np.log(v))**2 - 4*par**3*(-np.log(u))**par*(-np.log(v))**par*np.log(-np.log(v)) + 2*par**3*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(v))**2 - 4*par**3*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(v)) - 2*par**3*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par)*np.log(-np.log(v)) + 2*par**3*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par) - 3*par**3*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(v))**2 + 12*par**3*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(v)) - 2*par**3*(-np.log(v))**(2*par)*np.log((-np.log(u))**par + (-np.log(v))**par) - 4*par**3*(-np.log(v))**(2*par)*np.log(-np.log(v)) + par**2*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log(-np.log(u))**2 - 2*par**2*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log(-np.log(u)) - 4*par**2*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par)*np.log(-np.log(u)) + 4*par**2*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par) - 2*par**2*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(u))**2 + 8*par**2*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(u)) + par**2*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par)**2 + 6*par**2*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par)*np.log(-np.log(u)) - 12*par**2*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par) + 2*par**2*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u))**2 - 8*par**2*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u)) + 4*par**2*(-np.log(u))**(2*par)*np.log((-np.log(u))**par + (-np.log(v))**par) + 2*par**2*(-np.log(u))**(2*par)*np.log(-np.log(u)) + 2*par**2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log(-np.log(u))*np.log(-np.log(v)) - 2*par**2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log(-np.log(u)) - 2*par**2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log(-np.log(v)) - 4*par**2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par)*np.log(-np.log(u)) - 4*par**2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par)*np.log(-np.log(v)) + 8*par**2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par) - 4*par**2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(u))*np.log(-np.log(v)) + 8*par**2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(u)) + 8*par**2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(v)) + 2*par**2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par)**2 + 6*par**2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par)*np.log(-np.log(u)) + 6*par**2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par)*np.log(-np.log(v)) - 24*par**2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par) + 4*par**2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u))*np.log(-np.log(v)) - 8*par**2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u)) - 8*par**2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(v)) + 8*par**2*(-np.log(u))**par*(-np.log(v))**par*np.log((-np.log(u))**par + (-np.log(v))**par) + 2*par**2*(-np.log(u))**par*(-np.log(v))**par*np.log(-np.log(u)) + 2*par**2*(-np.log(u))**par*(-np.log(v))**par*np.log(-np.log(v)) + par**2*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log(-np.log(v))**2 - 2*par**2*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log(-np.log(v)) - 4*par**2*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par)*np.log(-np.log(v)) + 4*par**2*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par) - 2*par**2*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(v))**2 + 8*par**2*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(v)) + par**2*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par)**2 + 6*par**2*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par)*np.log(-np.log(v)) - 12*par**2*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par) + 2*par**2*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(v))**2 - 8*par**2*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(v)) + 4*par**2*(-np.log(v))**(2*par)*np.log((-np.log(u))**par + (-np.log(v))**par) + 2*par**2*(-np.log(v))**(2*par)*np.log(-np.log(v)) - 2*par*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log((-np.log(u))**par + (-np.log(v))**par)*np.log(-np.log(u)) + 2*par*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log((-np.log(u))**par + (-np.log(v))**par) + 2*par*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par)**2 + 4*par*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par)*np.log(-np.log(u)) - 8*par*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par) - 3*par*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par)**2 - 4*par*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par)*np.log(-np.log(u)) + 8*par*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par) - 2*par*(-np.log(u))**(2*par)*np.log((-np.log(u))**par + (-np.log(v))**par) - 2*par*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log((-np.log(u))**par + (-np.log(v))**par)*np.log(-np.log(u)) - 2*par*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log((-np.log(u))**par + (-np.log(v))**par)*np.log(-np.log(v)) + 4*par*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log((-np.log(u))**par + (-np.log(v))**par) + 4*par*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par)**2 + 4*par*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par)*np.log(-np.log(u)) + 4*par*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par)*np.log(-np.log(v)) - 16*par*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par) - 6*par*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par)**2 - 4*par*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par)*np.log(-np.log(u)) - 4*par*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par)*np.log(-np.log(v)) + 16*par*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par) - 4*par*(-np.log(u))**par*(-np.log(v))**par*np.log((-np.log(u))**par + (-np.log(v))**par) - 2*par*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log((-np.log(u))**par + (-np.log(v))**par)*np.log(-np.log(v)) + 2*par*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log((-np.log(u))**par + (-np.log(v))**par) + 2*par*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par)**2 + 4*par*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par)*np.log(-np.log(v)) - 8*par*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par) - 3*par*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par)**2 - 4*par*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par)*np.log(-np.log(v)) + 8*par*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par) - 2*par*(-np.log(v))**(2*par)*np.log((-np.log(u))**par + (-np.log(v))**par) + (-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log((-np.log(u))**par + (-np.log(v))**par)**2 - 2*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par)**2 + 2*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par)**2 + 2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log((-np.log(u))**par + (-np.log(v))**par)**2 - 4*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par)**2 + 4*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par)**2 + (-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log((-np.log(u))**par + (-np.log(v))**par)**2 - 2*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par)**2 + 2*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par)**2)/(par**4*((-np.log(u))**par + (-np.log(v))**par)**2*(par + ((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) - 1)**2)
        elif deriv == 'd2_u_v':
            res = (-np.log(u))**par*(-np.log(v))**par*(par - 1)*(2*par**3 + 5*par**2*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) - 3*par**2 + 4*par*((-np.log(u))**par + (-np.log(v))**par)**(2/par) - 5*par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) + par + ((-np.log(u))**par + (-np.log(v))**par)**(3/par) - 2*((-np.log(u))**par + (-np.log(v))**par)**(2/par) + 2*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)))/(u*v*((-np.log(u))**par + (-np.log(v))**par)**2*(par + ((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) - 1)**2*np.log(u)*np.log(v))
        elif deriv == 'd2_par_u':
            res = (-2*par**5*(-np.log(u))**par*(-np.log(v))**par*np.log(-np.log(u)) + 2*par**5*(-np.log(u))**par*(-np.log(v))**par*np.log(-np.log(v)) - par**4*(-np.log(u))**(2*par) - 5*par**4*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u)) + 5*par**4*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(v)) + 5*par**4*(-np.log(u))**par*(-np.log(v))**par*np.log(-np.log(u)) - 5*par**4*(-np.log(u))**par*(-np.log(v))**par*np.log(-np.log(v)) + par**4*(-np.log(v))**(2*par) - par**3*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u)) - 2*par**3*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) + 2*par**3*(-np.log(u))**(2*par) - 4*par**3*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(u)) + 4*par**3*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(v)) + 9*par**3*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u)) - 10*par**3*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(v)) - 4*par**3*(-np.log(u))**par*(-np.log(v))**par*np.log(-np.log(u)) + 4*par**3*(-np.log(u))**par*(-np.log(v))**par*np.log(-np.log(v)) + 2*par**3*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) - 2*par**3*(-np.log(v))**(2*par) - 2*par**2*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(u)) - par**2*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par) + par**2*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par) + 3*par**2*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u)) + par**2*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) - par**2*(-np.log(u))**(2*par) - par**2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log(-np.log(u)) + par**2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log(-np.log(v)) + 4*par**2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(u)) - 6*par**2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(v)) + par**2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par) - 4*par**2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u)) + 7*par**2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(v)) - par**2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) + par**2*(-np.log(u))**par*(-np.log(v))**par*np.log(-np.log(u)) - par**2*(-np.log(u))**par*(-np.log(v))**par*np.log(-np.log(v)) + par**2*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par) - 2*par**2*(-np.log(v))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) + par**2*(-np.log(v))**(2*par) - par*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log(-np.log(u)) + 2*par*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par) + 2*par*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(u)) - 3*par*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par) - 2*par*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u)) - par*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log(-np.log(v)) + 2*par*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par) + 2*par*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(v)) - 3*par*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par) - 2*par*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(v)) + (-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log((-np.log(u))**par + (-np.log(v))**par) - 2*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par) + 2*(-np.log(u))**(2*par)*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par) + (-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(3/par)*np.log((-np.log(u))**par + (-np.log(v))**par) - 2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par) + 2*(-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par))/(par**2*u*((-np.log(u))**par + (-np.log(v))**par)**2*(par + ((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) - 1)**2*np.log(u))
        else:
            assert deriv == 'd2_par_v'
            res = GumbelCopula.ll_deriv(par, v, u, deriv='d2_par_u')
        return res

    @staticmethod
    def hfun(par, u, v):
        # TODO: Add checks for the parameter
        res = -(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.exp(-((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)))/(v*((-np.log(u))**par + (-np.log(v))**par)*np.log(v))
        return res

    @staticmethod
    def inv_hfun(par, u, v):
        # TODO: Add checks for the parameter
        res = np.array([root_scalar(lambda xx: GumbelCopula.hfun(par, xx, v[i]) - u[i],
                                    bracket=[1e-12, 1-1e-12],
                                    method='brentq',
                                    xtol=1e-12, rtol=1e-12).root for i in range(len(u))])
        return res
