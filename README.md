# Monte_Carlo_von_Mises
The von Mises distribution admits the following density: ![von_Mises_density](https://i.imgur.com/yWeeFVl.png)

We simulated this distribution, for which the normalization constant is not known, using different Markov chain Monte Carlo (MCMC) approaches: 
1) Rejection Sampling
2) Random Walk Hastings-Metropolis (RWHM)

We also used Markov chain Monte Carlo Maximum likelihood Estimation (better known as MCMC-MLE) to predict the values of mu and kappa from this distribution, from observed data.
