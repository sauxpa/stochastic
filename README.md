# stochastic

### bivariate_copula
Code for simulating joint distribution based on a copula and empirical marginals.

### ito_diffusions_examples
Libraries for stochastic processes simulation and visualization including:
* Ito diffusion : Brownian motion, Geometric Brownian motion, Vasicek, CIR...
* Jump processes : Ito diffusion driven by a Levy process i.e with a jump component with a given intensity and jump size distribution
* Multidimensional processes, stochastic volatility diffusions (SABR...)
* Fractional Brownian motion, Karhunen-Loeve expansion, fractional diffusions

**To install** : pip install ito-diffusions
https://pypi.org/project/ito-diffusions/

<img src="./examples/ito_diffusions_examples/brownian_sheaf.png"
     style="float: left; margin-right: 10px;" />


### ou_fitting
Fit a Ornstein-Uhlenbeck process (potentially with Laplace jumps) on historical data using the generalized methods of moments on the characteristic functon.
