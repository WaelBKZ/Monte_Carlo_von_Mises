from tools import *
import matplotlib.pyplot as plt
from scipy.stats import vonmises


class ProjectModel:
    """
    Parent class to all our project models, gathering common methods and constructor for every Von Mises models.
    """

    def __init__(self, mu=0., kappa=3., x_init=0, proposal='gaussian', sig=2):
        # mu parameter
        self.mu = mu

        # kappa parameter
        self.kappa = kappa

        if self.__class__.__name__ == 'VonMisesRWHM':
            # the following objects are only built if the model is a Random Walk Hasting-Metropolis (RWHM) model;

            # initialisation value of the Markov chain;
            self.x_init = x_init

            # proposed distribution for the random walk; can be either 'gaussian' or 'uniform';
            self.proposal = proposal

            # standard deviation (gaussian) or size of the uniform distribution wished for our random walk proposal;
            self.sig = sig

            # number of acceptances and acceptation rate of the model, both initialized at 0;
            self.n_accept = 0
            self.acceptation_rate = 0

        # results of the simulation;
        self.results = []

    def hist(self, n=100_000):
        """
        Generates and prints the graph of n observations made under our model.

        :param int n:number of observations wished
        :return None:
        """
        if np.array_equal(self.results, []):
            print('You did not run the simulation yet. Please run {self}.simulate() first.')
            return None
        plt.gcf().clear()
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_axes([0.065, 0.1, 0.75, 0.8])

        ax.hist(self.results, bins=500, density=True, color='grey')
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
    """
    Accept-Reject simulation for Von Mises distribution.
    """

    def simulate(self, n=1):
        """
        Generates n observation(s) under our Von Mises Accept-Reject simulation.
        :param int n: number of observation(s) wished.
        :return list: list of observation(s), floats.
        """
        self.results = von_mises_unif(self.mu, self.kappa, n)
        return self.results


class VonMisesRWHM(ProjectModel):
    """
    Random Walk Hastings-Metropolis (RWMH) simulation for Von Mises distribution.
    """

    @staticmethod
    def proposal_step(proposal='gaussian', sig=2):
        """
    Computes the step of the random walk.
        :param str proposal: proposed function
        :param float sig: standard deviation (gaussian) or size of the uniform distribution wished for our random walk
        proposal
        :return float: the random walk step
        """
        if proposal == 'gaussian':
            return sig * np.random.randn()
        elif proposal == 'uniform':
            return sig * np.random.uniform(-1, 1)
        else:
            print('Wrong proposal')

    def iter(self, x):
        """
    Does an iteration of the RWHM simulation.
        :param float x: initial value
        :return float: new value of the Markov chain
        """
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
        """
        Generates n observation(s) under our Von Mises Random Walk Hastings-Metropolis simulation.
        :param int n: number of observation(s) wished.
        :return list: list of observation(s), floats.
        """
        self.n_accept = 0
        x = np.empty(n)
        x[0] = self.x_init
        for i in range(1, n):  # we realize (n-1) iterations of the Markov Chain
            x[i] = self.iter(x[i - 1])
        self.acceptation_rate = self.n_accept / n
        print(f'Acceptation rate: {self.acceptation_rate} %.')  # should be calibrated between 25% and 40%;
        self.results = x
        return self.results

    # Note pour le groupe : il faut peut-être penser à des moyens d'évaluer notre modèle ?


if __name__ == '__main__':
    mu = np.pi / 4
    kappa = 0.7
    n = 1_000_000
    print("Modèle d'acceptation rejet :")
    model = VonMisesAcceptReject(mu=mu, kappa=kappa)
    results = model.simulate(n=n)  # Génère un résultat de notre modèle Von Mises.
    model.hist(n=n)  # Génère un graphe à n observations.
    print(help(ProjectModel))
    #
    # print("Modèle de marche aléatoire Hastings-Metropolis:")
    # sig = 4
    # model = VonMisesRWHM(mu=mu, kappa=kappa, x_init=0, proposal='gaussian', sig=sig)
    # result = model.simulate(n=n)  # Génère un résultat de notre modèle Von Mises.
    # model.hist(n=n)  # Génère un graphe à n observations.
