import numpy as np
from scipy.linalg import sqrtm, norm
from scipy.spatial import distance_matrix
import cvxpy as cp


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
        Kbar_sqrt = sqrtm(Kbar)
        Kbar_new = np.sum([w*sqrtm(np.dot(Kbar_sqrt, np.dot(K, Kbar_sqrt))) for w, K in zip(weights, Ks)], axis=0)
        
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
        Kbar = cov2corr(Kbar)
            
    if verbose:
        return Kbar, errs
    else:
        return Kbar

    
def empirical_frechet_barycenter(cloud_points, 
                                 cloud_weights, 
                                 bar_weights=[],
                                 niter=10000,
                                 eps_abs=1e-5,
                                 eps_rel=1e-5,
                                 verbose=False,
                                ):
    """Calculation of Frechet barycenter w.r.t Wasserstein distance
    of discrete measures by solving the minimization problem.
    
    cloud_points: (T,N) point clouds,
    cloud_weights: list of k mass distributions,
    bar_weights: list of nonnegative weights summing to 1 in the barycenter computation;
        defaults to uniform weights,
    niter: maximum number of iterations,
    eps_abs: absolute tolerance,
    eps_rel: relative tolerance,
    verbose: print cvxpy info if True.
    """         
    k = cloud_weights.shape[1]
    T, N = cloud_points.shape

    if len(bar_weights) == 0:
        bar_weights = np.ones(k)/k
    
    # distance matrix on the point cloud
    D = distance_matrix(cloud_points, cloud_points)
    
    ### Add variables
    
    # optimal transport plan
    pi = []
    # for the epigraph formulation of the minimization problem
    t = []
    # barycenter weights
    mu = cp.Variable(T, nonneg = True)
    for i in range(k):
        pi.append(cp.Variable((T, T), nonneg = True))
        t.append(bar_weights[i]*cp.Variable(nonneg = True))

    obj = cp.Minimize(np.sum(t))

    ### Add constraints
    Cons = []
    for i in range(k):
        # epigraph formulation
        Cons.append( t[i] >= cp.sum(cp.multiply(D, pi[i])) )
        # marginal to barycenter
        Cons.append( (np.ones(T) @ pi[i]).T == mu)
        # marginal to cloud_weights
        Cons.append( (pi[i] @ np.ones(T)) == cloud_weights[:, i])

    prob = cp.Problem(obj, constraints= Cons)
    result = prob.solve(verbose=verbose, 
                        max_iter=niter, 
                        eps_abs=eps_abs,
                        eps_rel=eps_rel,
                       )
        
    return mu.value