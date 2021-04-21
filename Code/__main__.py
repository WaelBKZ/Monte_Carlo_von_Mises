from tools import *
import matplotlib.pyplot as plt
from scipy.stats import vonmises


class VonMises:
    def __init__(self, mu=0., kappa=3.):
        """
        Args:
            mu: Paramètre mu de la loi de Von Mises.
            kappa: Paramètre kappa de la loi de Von Mises.
        """
        self.mu = mu
        self.kappa = kappa

    def simulate(self, n=1):
        """
        Génère n observation(s) sous la loi de Von Mises.
        Args:
            n: nombre d'observations souhaitées.

        Returns: n observations sous la loi de Von Mises.
        """
        return von_mises_unif(self.mu, self.kappa, n)

    def hist(self, n=1000):
        """
        Génère le graphe de n observations sous la loi de Von Mises.
        Args:
            n: nombre d'observations souhaitées.
        """
        data = von_mises_unif(self.mu, self.kappa, n)
        plt.gcf().clear()
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_axes([0.065, 0.1, 0.75, 0.8])
        
        ax.hist(data, bins=500, density=True, color='grey')
        x = np.linspace(-np.pi, np.pi, 100)
        ax.plot(x, vonmises.pdf(x, self.kappa, loc=self.mu), 'r-', lw=1, label='theoretical')

        ax.text(3.85, 0.15,
                f'mu:         {self.mu:.3f}\nkappa:    {self.kappa:.3f}\nn:       {n:.1e}',
                style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
        ax.set_title('Von Mises simulation')
        plt.legend()
        plt.show()


# Note pour le groupe : il faut peut-être penser à des moyens d'évaluer notre modèle ?

if __name__ == '__main__':
    mu = np.pi/4
    kappa = 0.7
    n = 1_000_000
    model = VonMises(mu=mu, kappa=kappa)
    result = model.simulate(n=n)  # Génère un résultat de notre modèle Von Mises.

    model.hist(n=n)  # Génère un graphe à n observations.

