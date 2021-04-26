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

        X = np.hstack([X, x[(0 < u) & (u < 1)]])
        U = np.hstack([U, u[(0 < u) & (u < 1)]])
    return mean + std * X * np.sqrt(-2 * np.log(U) / U)


def von_mises_unif(mu=0., kappa=1., n=1):
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

        von_mises = np.hstack([von_mises, sample[uniform(n - len(von_mises)) <= val]])
    return von_mises


def von_mises_log(mu=0., kappa=1., n=1):
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

    x1, x2 = -0.4, 0.4
    a1, a2, a3, a4 = 2 * kappa / np.pi, log_f_(x1), log_f_(x2), -2 * kappa / np.pi

    b2, b3 = log_f(x1) - a2 * x1, log_f(x2) - a3 * x2

    z = (b3 - b2) / (a2 - a3)

    Q1 = np.exp(kappa) * (np.exp(a1 * -np.pi/2) - np.exp(a1 * -np.pi)) / a1
    Q2 = Q1 + np.exp(b2) * (np.exp(a2 * z) - np.exp(a2 * -np.pi/2)) / a2
    Q3 = Q2 + np.exp(b3) * (np.exp(a3 * np.pi/2) - np.exp(a3 * z)) / a3
    c = Q3 + np.exp(kappa) * (np.exp(a4 * np.pi) - np.exp(a4 * np.pi/2)) / a4

    def acceptance_val(X):
        sample1 = np.log(a1 * np.exp(-kappa) * X[X < Q1] + np.exp(a1 * -np.pi)) / a1
        val1 = np.exp(log_f(sample1) - a1 * sample1 - kappa)

        sample2 = np.log(a2 * np.exp(-b2) * (X[(Q1 <= X) & (X < Q2)] - Q1) + np.exp(a2 * -np.pi / 2)) / a2
        val2 = np.exp(log_f(sample2) - a2 * sample2 - b2)

        sample3 = np.log(a3 * np.exp(-b3) * (X[(Q2 <= X) & (X < Q3)] - Q2) + np.exp(a3 * z)) / a3
        val3 = np.exp(log_f(sample3) - a3 * sample3 - b3)

        sample4 = np.log(a4 * np.exp(-kappa) * (X[Q3 <= X] - Q3) + np.exp(a4 * np.pi / 2)) / a4
        val4 = np.exp(log_f(sample4) - a4 * sample4 - kappa)

        return np.hstack([sample1, sample2, sample3, sample4])[uniform(n) <= np.hstack([val1, val2, val3, val4])]

    # Compute a uniform on [-c, c]
    C = c * uniform(n=n)

    # Acceptance step
    von_mises = acceptance_val(C)

    # Keep computing until we have a sample on size n
    while len(von_mises) < n:
        C = c * uniform(n=n)
        von_mises = np.hstack([von_mises, acceptance_val(C)])
    return (von_mises + np.pi + mu) % (2*np.pi) - np.pi


        von_mises = np.concatenate((von_mises, sample[uniform(n) <= val]))
    return von_mises


def von_mises_density(x, mu=0., kappa=1.):
    """ Computes the density of a Von Mises distribution with parameters mu and kappa.
    Defined up to a constant multiplier.

    :param float x: point to be evaluated
    :param float mu: mu
    :param float kappa: kappa

    :return float: Von Mises density evaluated on x (up to a constant multiplier).
    """
    return np.exp(kappa * np.cos(x - mu)) * (-np.pi <= x <= np.pi)


def simulate(array):
    plt.gcf().clear()
    plt.hist(array, bins=200, density=True, color='grey')
    plt.legend()
    plt.show()
