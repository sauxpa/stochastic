from .base_copula import *

class GumbelCopula(BivariateCopula):
    def __init__(self,
                 X=None,
                 Y=None,
                 theta=None,
                ):
        super().__init__(X=X,
                         Y=Y,
                         theta=theta,
                        )
    @property
    def name(self):
        return 'Gumbel'
    
    def h(self, u=None, v=None):
        """h = dC/dv = P[U<u | V=v]
        """
        temp = ((-np.log(u)) ** self.theta + (-np.log(v)) ** self.theta)
        return -np.exp(-temp ** (1/self.theta)) * (temp ** (1/self.theta-1)) * (-np.log(v)) ** (self.theta)/(v*np.log(v))
        
    def C(self, u, v):
        return np.exp(-((-np.log(u)) ** self.theta + (-np.log(v)) ** self.theta) ** (1/self.theta))
    