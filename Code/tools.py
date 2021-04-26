import numpy as np
import matplotlib.pyplot as plt


def uniform(n=1):
    """ This function simulates a uniform distribution on [0, 1] """
    return np.random.uniform(0, 1, n)


def uniform_1(n=1):
    """ This function simulates a uniform distribution on [-1, 1] """
    return uniform(n=n) * 2 - 1


def uniform_pi(n=1):
    """ This function simulates a uniform distribution on [-pi, pi] """
    return uniform(n=n) * 2 * np.pi - np.pi


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

    :param float mu: mu
    :param float kappa: kappa
    :param int n: output size

    :return array: sample of a rv following a von mises distribution.
    """

    # Compute a uniform on [-pi, pi]
    sample = uniform_pi(n=n)

    # Compute the value for the rejection test
    val = np.exp(kappa * (np.cos(sample - mu) - 1))

    # Acceptance step
    von_mises = sample[uniform(n) <= val]

    # Keep computing until we have a sample on size n
    while len(von_mises) < n:
        sample = uniform_pi(n - len(von_mises))
        val = np.exp(kappa * (np.cos(sample - mu) - 1))

        von_mises = np.concatenate((von_mises, sample[uniform(n - len(von_mises)) <= val]))
    return von_mises


def von_mises_log(x1=-0.4, x2=0.4, mu=0., kappa=1., n=100):
    """
    This function simulates a von Mises distribution using the
    rejection sampler based on a uniform distribution on [-pi, pi]
    as proposal distribution.

    :param float mu: mu
    :param float kappa: kappa
    :param int n: output size

    :return array: sample of a rv following a von mises distribution.
    """

    def log_f(x):
        return kappa * np.cos(x)

    def log_f_(x):
        return - kappa * np.sin(x)

    a1 = 2 * kappa / np.pi
    a2 = log_f_(x1)
    a3 = log_f_(x2)
    a4 = -a1

    b1 = kappa
    b2 = log_f(x1) - a2 * x1
    b3 = log_f(x2) - a3 * x2
    b4 = kappa

    z0 = -np.pi
    z1 = -np.pi / 2
    z2 = (b3 - b2) / (a2 - a3)
    z3 = np.pi / 2
    z4 = np.pi

    Q1 = np.exp(b1) * (np.exp(a1 * z1) - np.exp(a1 * z0)) / a1
    Q2 = Q1 + np.exp(b2) * (np.exp(a2 * z2) - np.exp(a2 * z1)) / a2
    Q3 = Q2 + np.exp(b3) * (np.exp(a3 * z3) - np.exp(a3 * z2)) / a3
    c = Q3 + np.exp(b4) * (np.exp(a4 * z4) - np.exp(a4 * z3)) / a4

    def acceptance_val(x):
        """
        This function computes the acceptance value.
        """

        if x < Q1:
            z = np.log(a1 * np.exp(-b1) * x + np.exp(a1 * z0)) / a1
            return [z, np.exp(log_f(z) - a1 * z - b1)]
        elif x < Q2:
            z = np.log(a2 * np.exp(-b2) * (x - Q1) + np.exp(a2 * z1)) / a2
            return [z, np.exp(log_f(z) - a2 * z - b2)]
        elif x < Q3:
            z = np.log(a3 * np.exp(-b3) * (x - Q2) + np.exp(a3 * z2)) / a3
            return [z, np.exp(log_f(z) - a3 * z - b3)]
        else:
            z = np.log(a4 * np.exp(-b4) * (x - Q3) + np.exp(a4 * z3)) / a4
            return [z, np.exp(log_f(z) - a4 * z - b4)]

    # Compute a uniform on [-c, c]
    C = c * uniform(n=n)

    # Compute the value for the rejection test
    res = np.array([acceptance_val(x) for x in C])
    sample, val = res[:, 0], res[:, 1]

    # Acceptance step
    von_mises = sample[uniform(n) <= val]

    # Keep computing until we have a sample on size n
    while len(von_mises) < n:
        C = c * uniform(n=n)
        res = np.array([acceptance_val(x) for x in C])
        sample, val = res[:, 0], res[:, 1]

        von_mises = np.concatenate((von_mises, sample[uniform(n) <= val]))
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
