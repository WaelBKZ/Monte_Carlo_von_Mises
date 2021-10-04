# Monte_Carlo_von_Mises
The von Mises distribution admits the following density: ![von_Mises_density](https://i.imgur.com/yWeeFVl.png)

We simulated this distribution, for which the normalization constant is not known, using different Markov chain Monte Carlo (MCMC) approaches: 
1) Rejection Sampling
2) Random Walk Hastings-Metropolis (RWHM)

We have implemented evaluation methods for the performance of our algorithms. We have also coded a fitting method that optimizes the choice of hyperparameters for the algorithm.  

We have incorporated Markov chain Monte Carlo Maximum likelihood Estimation (better known as MCMC-MLE) in order to predict the values of mu and kappa in this distribution from observed data.
