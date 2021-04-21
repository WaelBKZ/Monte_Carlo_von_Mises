import numpy as np
from scipy.stats import vonmises
import matplotlib.pyplot as plt
from math import *


def uniform(n=1):
    """ This function simulates a uniform distribution on [0, 1] """
    return np.random.uniform(0, 1, n)


def uniform_1(n=1):
    """ This function simulates a uniform distribution on [-1, 1] """
    return uniform(n=n) * 2 - 1


def uniform_pi(n=1):
    """ This function simulates a uniform distribution on [-pi, pi] """
    return uniform(n=n) * 2 * np.pi - np.pi


def cauchy(mean=0., std=1., n=1):
    """ This function simulates a cauchy distribution """
    return mean + std * np.tan(np.pi * (uniform(n=n) - 0.5))


def normal(mean=0., std=1., n=1):
    """ This function simulates a normal distribution """

    X, Y = uniform_1(n=n), uniform_1(n=n)
    U = X ** 2 + Y ** 2

    X, U = X[(0 < U) & (U < 1)], U[(0 < U) & (U < 1)]
    while len(X) < n:
        x, y = uniform_1(n=n - len(X)), uniform_1(n=n - len(X))
        u = x ** 2 + y ** 2

        X = np.concatenate((X, x[(0 < u) & (u < 1)]))
        U = np.concatenate((U, u[(0 < u) & (u < 1)]))
    return mean + std * X * np.sqrt(-2 * np.log(U) / U)


def von_mises_unif(mu=0., kappa=1., n=100):
    """ This function simulates a von Mises distribution using the
    rejection sampler based on a uniform distribution on [-pi, pi]
    as proposal distribution.

    In this case, because g is constant on [-pi, pi] the optimal value for the
    rejection sampling is the maximum value of the f on [-pi, pi].

    Overall, we have to :
        - simulate a uniform variable on [-pi, pi].
        - compute the value of exp(kappa * cos(theta - mu)) / exp(kappa).
        - simulate a uniform variable on [0, 1].
        - reject the value of theta if u > exp(kappa * (cos(theta - mu)- 1))

    :param float mu: mu
    :param float kappa: kappa
    :param int n: output size

    :return array: sample of a rv following a von mises distribution.
    """

    # Compute a uniform on [-pi, pi]
    sample = uniform_pi(n=n)

    # Compute the value for the rejection test
    val = np.exp(kappa * (np.cos(sample - mu) - 1))

    # Compute a uniform on [0, 1]
    unif = uniform(n)

    # Reject the values
    von_mises = sample[unif <= val]

    # Keep computing until we have a sample on size n
    while len(von_mises) < n:
        sample = uniform_pi(n - len(von_mises))
        val = np.exp(kappa * (np.cos(sample - mu) - 1))
        unif = uniform(n - len(von_mises))

        von_mises = np.concatenate((von_mises, sample[unif <= val]))
    return von_mises


def von_mises_density(x, mu=0, kappa=3):
    """ Computes the density of a Von Mises distribution with parameters mu and kappa.
    Defined up to a constant multiplier.

    :param float x: point to be evaluated
    :param float mu: mu
    :param float kappa: kappa

    :return float: Von Mises density evaluated on x (up to a constant multiplier).
    """
    return np.exp(kappa * np.cos(x - mu)) * (-np.pi <= x <= np.pi)
