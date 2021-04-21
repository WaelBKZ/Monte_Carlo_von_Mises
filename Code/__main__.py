from tools import *
import matplotlib.pyplot as plt
from scipy.stats import vonmises


class ProjectModel:
    # Classe qui servira de classe parente à toutes nos autres classes de modèles, qui concentrera les méthodes
    # qu'on utilisera à différentes reprises. N'a pas vocation à être utilisée seule.
    def __init__(self, mu=0., kappa=3., x_init=0, proposal='gaussian', sig=2):
        self.mu = mu
        self.kappa = kappa
        if self.__class__.__name__ == 'VonMisesRWHM':
            self.x_init = x_init  # Valeur d'initialisation de la chaîne
            self.proposal = proposal  # proposal = 'gaussian' ou 'uniform' : fonction proposée pour la marche aléatoire
            self.sig = sig  # écart-type voulu pour la fonction de marche aléatoire
            self.n_accept = 0  # Nombre d'acceptation de l'algorithme de Metropolis
            self.acceptation_rate = 0  # Taux d'acceptation de l'algorithme de Metropolis
        self.data = []  # Sortie de la simulation

    def hist(self, n=100_000):
        """
        Génère le graphe de n observations sous la loi de Von Mises.
        Args:
            n: nombre d'observations souhaitées.
        """
        plt.gcf().clear()
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_axes([0.065, 0.1, 0.75, 0.8])

        ax.hist(self.data, bins=500, density=True, color='grey')
        x = np.linspace(-np.pi, np.pi, 100)
        ax.plot(x, vonmises.pdf(x, self.kappa, loc=self.mu), 'r-', lw=1, label='theoretical')

        ax.text(3.85, 0.15,
                f'mu:         {self.mu:.3f}\nkappa:    {self.kappa:.3f}\nn:       {n:.1e}',
                style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
        ax.set_title(f'Von Mises simulation : {self.__class__.__name__}.')
        plt.legend()
        plt.show()


class VonMisesAcceptReject(ProjectModel):
    # Modèle d'acceptation-rejet de Von Mises
    def simulate(self, n=1):
        """
        Génère n observation(s) sous la loi de Von Mises.
        Args:
            n: nombre d'observations souhaitées.

        Returns: n observations sous la loi de Von Mises.
        """
        self.data = von_mises_unif(self.mu, self.kappa, n)
        return self.data


class VonMisesRWHM(ProjectModel):
    # A simulation of Von Mises law following a Random Walk Hastings-Metropolis simulation
    @staticmethod
    def proposal_step(proposal='gaussian', sig=2):
        if proposal == 'gaussian':
            return sig * np.random.randn()
        elif proposal == 'uniform':
            return sig * np.random.uniform(-1, 1)
        else:
            print('Wrong proposal')

    def iter(self, x):
        # réalise une itération de l'algorithme
        proposal_step = self.proposal_step(self.proposal, self.sig)
        y = x + proposal_step
        r = von_mises_density(y, mu=self.mu, kappa=self.kappa) / von_mises_density(x, mu=self.mu, kappa=self.kappa)
        u = np.random.rand()
        if u < r:
            self.n_accept += 1
            return y
        else:
            return x

    def simulate(self, n=100_000):
        # proposal = 'gaussian' ou 'uniform'
        self.n_accept = 0  # Nombre d'acceptation de l'algorithme de Metropolis
        x = np.empty(n)
        x[0] = self.x_init
        for i in range(1, n):  # On fait les n itérations de la chaîne de Markov
            x[i] = self.iter(x[i - 1])
        self.acceptation_rate = self.n_accept / n
        print(f'Acceptation rate: {self.acceptation_rate} %.')
        self.data = x
        return self.data

    # Note pour le groupe : il faut peut-être penser à des moyens d'évaluer notre modèle ?


if __name__ == '__main__':
    mu = np.pi / 4
    kappa = 0.7
    n = 1_000_000
    print("Modèle d'acceptation rejet :")
    model = VonMisesAcceptReject(mu=mu, kappa=kappa)
    result = model.simulate(n=n)  # Génère un résultat de notre modèle Von Mises.
    model.hist(n=n)  # Génère un graphe à n observations.
    print()

    print("Modèle de marche aléatoire Hastings-Metropolis:")
    sig = 4
    model = VonMisesRWHM(mu=mu, kappa=kappa, x_init=0, proposal='gaussian', sig=sig)
    result = model.simulate(n=n)  # Génère un résultat de notre modèle Von Mises.
    model.hist(n=n)  # Génère un graphe à n observations.

