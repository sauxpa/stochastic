import numpy as np
from scipy.linalg import sqrtm, norm


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


def frechet_barycenter_corr(Ks,
                            weights=[],
                            force_corr=False,
                            force_real=True,
                            niter=100, 
                            tol=1e-8, 
                            ord=2, 
                            verbose=False,
                           ):
    """Compute iteratively the Frechet barycenter of
    Gaussian measures N(0, K) for K in Ks, w.r.t 2-Wasserstein distance.
    K^{t+1} = ( \sum_K (K^t * K * K^t) ** 0.5 ) ** 0.5
    
    Ks: list of matrices to average,
    weights: array of length len(Ks) of nonnegative numbers summing to 1; 
        defaults to uniform weights,
    force_corr: if True, renormalize the final iterate to make a correlation matrix
    (by default the iterations produce covariance matrices, but not necessarily correlation ones,
    even when all matrices in Ks are correlation matrices),
    force_real : if True, remove imaginary part at each iteration; these can appear due to
    numerical instability when taking the square root (usually for low/negative correlations).
    niter: maximum number of iterations,
    tol: stopping threshold on the distance between consecutive updates,
    ord: order of the norm to compute distance for tol (default: L2);
        if ord='wasserstein', use the 2-Wasserstein distance between Gaussians,
    verbose: if True, store successive errors and returns them.
    """
    # initialize barycenter to the Euclidean average of Ks.
    Kbar = np.mean(Ks, axis=0)
    if verbose:
        errs = []

    if len(weights) == 0:
        weights = np.ones(len(Ks))*1/len(Ks)
    else:
        weights = np.array(weights)
        
    for _ in range(niter):
        Kbar_new = np.sum([w*sqrtm(np.dot(sqrtm(Kbar), np.dot(K, sqrtm(Kbar)))) for w, K in zip(weights, Ks)], axis=0)
        if force_real:
            Kbar_new = np.real(Kbar_new)
        
        if ord == 'wasserstein':
            # 2-Wasserstein distance between centered Gaussian
            # is the Frobenius norm of the difference of the square
            # root of their covariance matrices
            err = norm(Kbar_new-Kbar, ord='fro')
        else:
            err = norm(Kbar_new-Kbar, ord=ord)
        
        if verbose:
            errs.append(err)
        
        Kbar = Kbar_new
        
        if err < tol:
            break
    
    if force_corr:
        D_inv = 1/np.sqrt(np.diag(Kbar))
        Kbar = np.multiply(D_inv, np.multiply(Kbar, D_inv))
            
    if verbose:
        return Kbar, errs
    else:
        return Kbar