from tools import *
import matplotlib.pyplot as plt
from scipy.stats import vonmises


class VonMises:
    def __init__(self, mu=0, kappa=3):
        """
        Args:
            mu: Paramètre mu de la loi de Von Mises.
            kappa: Paramètre kappa de la loi de Von Mises.
        """
        self.mu = mu
        self.kappa = kappa

    def generate(self, n=1):
        """
Génère n observation(s) sous la loi de Von Mises.
        Args:
            n: nombre d'observations souhaitées.

        Returns: n observations sous la loi de Von Mises.
        """
        return von_mises_unif(self.mu, self.kappa, n)

    def generate_graph(self, n=1000):
        """
Génère le graphe de n observations sous la loi de Von Mises.
        Args:
            n: nombre d'observations souhaitées.
        """
        data = von_mises_unif(self.mu, self.kappa, n)
        fig, ax = plt.subplots(1, 1)
        ax.hist(data, bins=500, density=True)
        x = np.linspace(vonmises.ppf(0.01, self.kappa, loc=self.mu), vonmises.ppf(0.99, self.kappa, loc=self.mu), 100)
        ax.plot(x, vonmises.pdf(x, self.kappa, loc=self.mu), 'r-', lw=1, label='theoretical')
        ax.set_title('Von Mises simulation')
        plt.legend()
        plt.show()


#Note pour le groupe : il faut peut-être penser à des moyens d'évaluer notre modèle ?

if __name__ == '__main__':
    mu = 0
    kappa = 3
    n = 100_000
    model = VonMises(mu, kappa)
    result = model.generate()  # Génère un résultat de notre modèle Von Mises.
    print(result)
    model.generate_graph(n)  # Génère un graphe à n observations.

