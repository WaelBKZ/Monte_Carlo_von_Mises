import numpy as np


def uniform_pi(n=1):
    """ This function simulates a uniform distribution on [-pi, pi] """
    return np.random.uniform(-np.pi, np.pi, n)


def uniform(n=1):
    """ This function simulates a uniform distribution on [0, 1] """
    return np.random.uniform(0, 1, n)


def normal(mean=0, std=1, n=1):
    """ This function simulates a normal distribution """
    return np.random.normal(mean, std, n)


def von_mises_unif(mu=0, kappa=1, n=1):
    """ This function simulates a von Mises distribution using the
    rejection sampler based on a uniform distribution on [-pi, pi]
    as proposal distribution.

    In this case :
        - g(theta) = 1 / (2*pi)
        - f(theta) = 1 / Z(kappa) * exp(kappa * cos(x - mu))
    Because g is constant on [-pi, pi] the optimal value for the
    rejection sampling is the maximum value of the f on [-pi, pi].

    Overall, we have to :
        - simulate a uniform variable on [-pi, pi].
        - compute the value of exp(kappa * cos(theta - mu)) / exp(kappa).
        - simulate a uniform variable on [-1, 1].
        - reject the value of theta if u > exp(kappa * (cos(theta - mu)- 1))

    """

    # Compute a uniform on [-pi, pi]
    unif_pi = uniform_pi(n)

    # Compute the value for the rejection test
    val = np.exp(kappa * (np.cos(unif_pi - mu) - 1))

    # Compute a uniform on [0, 1]
    unif = uniform(n)

    # Reject the values
    von_mises = unif_pi[unif <= val]

    # Keep computing until we have a sample on size n
    while len(von_mises) < n:
        unif_pi = uniform_pi(n - len(von_mises))
        val = np.exp(kappa * (np.cos(unif_pi - mu) - 1))
        unif = uniform(n - len(von_mises))

        von_mises = np.concatenate((von_mises, unif_pi[unif <= val]))
    return von_mises

