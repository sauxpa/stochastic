from scipy.stats import norm
from .base_copula import *

class GaussianCopula(BivariateCopula):
    def __init__(self,
                 X=None,
                 Y=None,
                 theta=None,
                ):
        super().__init__(X=X,
                         Y=Y,
                         theta=theta, # theta is the Gaussian correlation here 
                        )
    @property
    def name(self):
        return 'Gaussian'

    @property
    def cov(self):
        return np.array([[1.0, self.theta], [self.theta, 1.0]])
        
    def h(self, u=None, v=None):
        """No need here.
        """
        pass
    
    def C(self, u=None, v=None):
        """No need here.
        """
        pass
    
    def simulate(self, size=1):
        """Overwrite simulate function since for Gaussian copula there is an exact, faster
        way to sample from the copula distribution (no need to invert the h function, the
        usual Cholesky decomposition will do).
        """
        # bivariate Cholesky decomposition
        U = np.random.randn(size, 2)
        V = U[:, 1]
        U = U[:, 0]
        
        # after this step, [U,V] is sampled from a centered Gaussian vector
        # with variance 1 and correlation self.theta
        U = self.theta * V + np.sqrt(1-self.theta ** 2 ) * U
        
        # transform to uniform
        U, V = norm.cdf(U), norm.cdf(V)
        
        return self.inv_ecdf_X(U), self.inv_ecdf_Y(V)
