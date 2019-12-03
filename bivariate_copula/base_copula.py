import numpy as np
import abc
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d
from scipy.optimize import brentq

class BivariateCopula(abc.ABC):
    """
    Bases class for bivariate copulas.
    """
    def __init__(self,
                X=None,
                Y=None,
                theta=None,
                eps=1e-12
                ):
        # Data
        self._X = X
        self._Y = Y

        # Archimedian copula parameter
        self._theta = theta

        # numerical tolerance
        self._eps = eps

    @property
    @abc.abstractmethod
    def name(self):
        pass
            
    @property
    def X(self):
        return self._X
    @X.setter
    def X(self, new_X):
        self._X = new_X

    @property
    def Y(self):
        return self._Y
    @Y.setter
    def Y(self, new_Y):
        self._Y = new_Y

    @property
    def theta(self):
        return self._theta
    @theta.setter
    def theta(self, new_theta):
        self._theta = new_theta

    @property
    def eps(self):
        return self._eps
    @eps.setter
    def eps(self, new_eps):
        self._eps = new_eps

    @property
    def ecdf_X(self):
        return ECDF(self.X)

    @property
    def ecdf_Y(self):
        return ECDF(self.Y)

    def inv_ecdf(self, data):
        """Compute the empirical inverse cdf.
        """
        ecdf = ECDF(data)
        slope_changes = sorted(set(data))
        # add extrapolation boundaries
        slope_changes = np.concatenate([[-np.inf], slope_changes, [np.inf]])

        sample_edf_values_at_slope_changes = [ecdf(t) for t in slope_changes]
        inverted_edf = interp1d(sample_edf_values_at_slope_changes, slope_changes)

        return inverted_edf

    @property
    def inv_ecdf_X(self):
        return self.inv_ecdf(self.X)

    @property
    def inv_ecdf_Y(self):
        return self.inv_ecdf(self.Y)

    @abc.abstractmethod
    def C(self, u, v):
        """(u,v) --> C(u,v) copula cdf function.
        """
        pass

    @abc.abstractmethod
    def h(self, u=None, v=None):
        """h = dC/dv = P[U<u | V=v]
        """
        pass

    def simulate(self, size=1):
        """Simulate joint distribution (A, B) such that
        A and B have same empirical marginals as X and Y
        and their joint law is given by the copula.
        """

        T = np.random.uniform(size=(size, 2))

        # Sample for copula ditribution :
        # 1) generate independant uniform random variables
        # 2) Applay the pseudo-inverse of the h function to the first one
        # where the h function is the conditional cdf of the second variable w.r.t the first.
        U = np.array([brentq(lambda u: self.h(u, t[1]) - t[0], self.eps, 1.0-self.eps) for t in T])
        V = T[:, 1]

        return self.inv_ecdf_X(U), self.inv_ecdf_Y(V)
