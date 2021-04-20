from tools import *
import matplotlib.pyplot as plt
from scipy.stats import vonmises

if __name__ == '__main__':
    mu = 0
    kappa = 3
    n = 100_000

    data = von_mises_unif(mu=mu, kappa=kappa, n=n)
    
    fig, ax = plt.subplots(1, 1)
    ax.hist(data, bins=500, density=True)

    x = np.linspace(vonmises.ppf(0.01, kappa, loc=mu), vonmises.ppf(0.99, kappa, loc=mu), 100)
    ax.plot(x, vonmises.pdf(x, kappa, loc=mu), 'r-', lw=1, label='theoretical')
    ax.set_title('Von Mises simulation')
    plt.legend()
    plt.show()


