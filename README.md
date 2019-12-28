# stochastic

### bivariate_copula
Code for simulating joint distribution based on a copula and empirical marginals.

### corr_lib
Utility lib for correlation matrix manipulation and optimal transport metric.

### ito_diffusions and ito_diffusions_examples
Libraries for stochastic processes simulation and visualization including:
* Ito diffusion : Brownian motion, Geometric Brownian motion, Vasicek, CIR...
* Jump processes : Ito diffusion driven by a Levy process i.e with a jump component with a given intensity and jump size distribution
* Multidimensional processes, stochastic volatility diffusions (SABR...)
* Fractional Brownian motion, Karhunen-Loeve expansion, fractional diffusions

### optimal_transport_corr
Study Frechet barycenter w.r.t Wasserstein distance for correlation matrix metric using corr_lib module:
* compute the metric induced on the covariance manifold by the 2-Wasserstein distance on Gaussian measures,
* compute geodesic on the covariance manifold using weighted Frechet barycenter, and show that it is not a geodesic on the correlation manifold (matrices on the geodesic path between two correlation matrices within the covariance manifold need not be correlation matrices themselves).
* compute the metric induced on empirical correlations by a Wasserstein distance on the discrete measures, by solving Frechet barycenter problem as a linear program using CVXPY.


### ou_fitting
Fit a Ornstein-Uhlenbeck process (potentially with Laplace jumps) on historical data using the generalized methods of moments on the characteristic functon.
