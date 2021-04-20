import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def uniform_pi(n):
    """ This function simulates a random variable on [-pi, pi]. """
    return np.random.uniform(-1, 1, n) * np.pi
