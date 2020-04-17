import numpy as np
from functools import lru_cache
from tqdm import tqdm


class FastChordalSLEHalfPlane():
    """Simulate chordal Schramm Loewner Evolution
    random planar curve in the complex half-plane.
    Chordal here means the curve goes from a boundary
    point x0 to the boundary point at infinity.
    (in practice the process is stopped at T rather
    than infinity).
    """
    def __init__(self,
                 n: int = 1000,
                 kappa: float = 0.0,
                 T: float = 1.0,
                 x0: complex = 0j,
                 ):
        self._n = n
        self._kappa = kappa
        self._T = T
        self._x0 = x0

    @property
    def n(self) -> int:
        return self._n

    @n.setter
    def n(self, new_n: int) -> None:
        type(self).scheme_step.fget.cache_clear()
        self._n = new_n

    @property
    def kappa(self) -> float:
        return self._kappa

    @kappa.setter
    def kappa(self, new_kappa: float) -> None:
        type(self).alpha_minus.fget.cache_clear()
        type(self).alpha_plus.fget.cache_clear()
        self._kappa = new_kappa

    @property
    def T(self) -> float:
        return self._T

    @T.setter
    def T(self, new_T: float) -> None:
        type(self).scheme_step.fget.cache_clear()
        self._T = new_T

    @property
    def x0(self) -> complex:
        return self._x0

    @x0.setter
    def x0(self, new_x0: complex) -> None:
        self._x0 = new_x0

    @property
    @lru_cache(maxsize=None)
    def alpha_minus(self) -> float:
        return 0.5 * (1 - np.sqrt(self.kappa / (16 + self.kappa)))

    @property
    @lru_cache(maxsize=None)
    def alpha_plus(self) -> float:
        return 0.5 * (1 + np.sqrt(self.kappa / (16 + self.kappa)))

    @property
    @lru_cache(maxsize=None)
    def scheme_step(self) -> float:
        return self.T / self.n

    def _h(self, z: complex, alpha: float) -> complex:
        return (
            z + 2 * np.sqrt(
                (self.scheme_step * (1 - alpha))/alpha)
            ) ** (1 - alpha) * (
                z - 2 * np.sqrt(
                    (self.scheme_step * alpha) / (1 - alpha)
                    )
                ) ** alpha

    def simulate(self):
        """Simulate by recursively computing the sequence
        of iterates, drawing the sign of alpha at random
        at each step.
        """
        z = np.array([self.x0])
        with tqdm(total=self.n) as pbar:
            for i in range(1, self.n+1):
                if np.random.rand() > 0.5:
                    z = np.concatenate(
                        [
                            [0.0],
                            self._h(z, self.alpha_plus)
                        ]
                    )
                else:
                    z = np.concatenate(
                        [
                            [self.x0],
                            self._h(z, self.alpha_minus)
                        ]
                    )
                pbar.update(1)
        return np.real(z + self.x0), np.imag(z + self.x0)
