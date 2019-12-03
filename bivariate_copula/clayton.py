from .base_copula import *

class ClaytonCopula(BivariateCopula):
    def __init__(self,
                 X=None,
                 Y=None,
                 theta=None,
                ):
        super().__init__(X=X,
                         Y=Y,
                         theta=theta,
                        )

    def h(self, u=None, v=None):
        """h = dC/dv = P[U<u | V=v]
        """
        return v ** (-self.theta-1) * self.A(u, v) ** (-1-1/self.theta)

    def A(self, u, v):
        """Helper function to define Clayton copula.
        """
        return np.max([u ** (-self.theta) + v ** (-self.theta) - 1.0, 0.0])

    def C(self, u, v):
        return self.A(u, v) ** (-1/self.theta)
