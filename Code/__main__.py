from tools import *
import matplotlib.pyplot as plt
from scipy.stats import vonmises
from statsmodels.graphics.tsaplots import plot_acf


class ProjectModel:
    """
    Parent class to all our project models, gathering common methods and constructor for every Von Mises models.
    """

    def __init__(self, mu=0., kappa=3., x_init=0., proposal='gaussian', sig=2.):
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

        # number of observations the simulation will predict; 0 before assignation;
        self.number_observations = 0

        # results of the simulation;
        self.results = []

    def hist(self):
        """
        Generates and prints the graph of the observations made under the simulation of our model.
        User needs to run {self}.simulation(n) method first.
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
                f'mu:         {self.mu:.3f}\nkappa:    {self.kappa:.3f}\nn:       {self.number_observations:.1e}',
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
        self.number_observations = n
        self.results = von_mises_unif(self.mu, self.kappa, n)
        return self.results


class VonMisesRWHM(ProjectModel):
    """
    Random Walk Hastings-Metropolis (RWHM) simulation for Von Mises distribution.
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
        self.number_observations = n
        self.n_accept = 0
        x = np.empty(n)
        x[0] = self.x_init
        for i in range(1, n):  # we realize (n-1) iterations of the Markov Chain
            x[i] = self.iter(x[i - 1])
        self.acceptation_rate = self.n_accept / n
        print(f'Acceptation rate: {self.acceptation_rate} %.')  # should be calibrated between 25% and 40%;
        self.results = x
        return self.results

    def graph_chain(self, n_points=1000):
        """
        Draw the the Markov chain. A sanity-check for this MCMC simulation is to have the graph looks 'random' : i.e. a
        randoms series where no particular pattern appears.
        :param int n_points: number of points to display
        :return None: returns the graph of the chain.
                """
        plt.plot(self.results[:n_points])
        plt.show()

    def graph_autocorrelation(self):
        """
        Draw the autocorrelations of the Markov chain. A sanity-check for this MCMC simulation is to have the
        autocorrelations plummet rapidly.
        :return None: returns the graph of the autocorrelations.
                """
        plot_acf(self.results)
        plt.show()

    # Note pour le groupe : il faut peut-être penser à des moyens d'évaluer notre modèle ?


if __name__ == '__main__':
    # Parameters common to every model:
    mu = np.pi / 4
    kappa = 0.7
    n = 100_000

    ### ACCEPT-REJECT:
    print("Accept-Reject simulation:")
    model = VonMisesAcceptReject(mu=mu, kappa=kappa)
    model.simulate(n=n)  # generates n observations under the Accept-Reject simulation;
    model.hist()  # generates the histogram of the above observations;



    ### RANDOM WALK HASTINGS-METROPOLIS:

    #Parameters exclusive to RWHM:
    x_init = 0
    proposal = 'gaussian'
    #proposal = 'uniform'
    sig = 5.5

    print("\nRandom Walk Hastings-Metropolis simulation:")
    model = VonMisesRWHM(mu=mu, kappa=kappa, x_init=x_init, proposal=proposal, sig=sig)
    model.simulate(n=n)  # generates n observations under the Random Walk Hastings-Metropolis simulation;
    model.hist()  # generates the histogram of the above observations;



    ## SANITY CHECK FOR RWHM:
    model.graph_autocorrelation()
    model.graph_chain(n_points=500)
    # implementation of burn-in sanity-check doesn't have much value there since the distribution is very narrow
    # there (between [-pi,pi]), thus the simulation cannot really 'get lost'.