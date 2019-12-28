import numpy as np


def cov2corr(cov):
    D_inv = 1/np.sqrt(np.diag(cov))
    return np.multiply(D_inv, np.multiply(cov, D_inv))


def weighted_correlation(cloud, weights, force_corr=True):
    cov = np.cov(cloud.T, aweights=weights)
    if force_corr:
        return cov2corr(cov)
    else:
        return cov

    
def random_corr_matrix(n, id_mixing = 0, gen=np.random.rand):
    """
    Draws a random correlation matrix (symmetric positive semidefinite
    from a seed matrix sampled from gen).
    This procedure tends to produce high correlation, id_mixing pulls
    the resulting matrix closer to the independent variables case
    (i.e identity correlation matrix).
    """
    C = gen(n, n)
    C = np.dot(C, C.T)
    std_dev = np.sqrt(np.diag(C))
    std_dev_inv = np.diag(1/std_dev)
    corr = np.matmul(std_dev_inv, np.matmul(C, std_dev_inv))
    return (1-id_mixing)*corr+id_mixing*np.eye(n)


def single_corr_matrix(n, rho):
    """
    Returns a correlation matrix where all variable have same correlation rho
    """
    if rho < 0.0 or rho >1.0:
        raise NameError('Rho is a correlation and therefore must be between 0 and 1')

    return np.ones(n)*rho+np.diag(np.ones((n,)))*(1-rho)
