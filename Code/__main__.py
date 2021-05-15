import logging
import matplotlib.pyplot as plt
from scipy.stats import vonmises
from statsmodels.graphics.tsaplots import plot_acf
from tools import *


class ProjectModel:
    """
    Parent class to all our project models classes, gathering common methods and constructor for every Von Mises models.
    """

    def __init__(self, mu=0., kappa=1., x_init=0., proposal='cauchy', proposal_RWHM='gaussian', sig=2.):
        # mu parameter
        self.mu = mu

        # kappa parameter
        self.kappa = kappa

        # propsed distribution for the rejection sampling; can be either 'cauchy' or 'uniform'
        self.proposal = proposal

        if self.__class__.__name__ == 'VonMisesRWHM':
            # the following objects are only built if the model is a Random Walk Hasting-Metropolis (RWHM) model;

            # initialisation value of the Markov chain;
            self.x_init = x_init

            # proposed distribution for the random walk; can be either 'gaussian' or 'uniform';
            self.proposal_RWHM = proposal_RWHM

            # standard deviation (gaussian) or size of the uniform distribution wished for our random walk proposal;
            self.sig = sig

            # number of acceptances and acceptance rate of the model, both initialized at 0;
            self.n_accept = 0
            self.acceptance_rate = 0

        # number of observations the simulation will predict; 0 before assignation;
        self.number_observations = 0

        # results of the simulation;
        self.results = None

    def describe_mu(self, save=False):
        """ This function plots the density of a von-Mises distribution for different values of mu """

        mu = [-np.pi/2, 0, np.pi/2]

        fig, ax = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
        i = 0
        x = np.linspace(-np.pi, np.pi, 300)
        for val in mu:
            ax[i].plot(x, vonmises.pdf(x, self.kappa, loc=val), 'r-', lw=1)  # On trace
            ax[i].set_ylim([0., None])  # On impose que l'axe des ordonnées commence par la valeur 0.
            ax[i].set_xticks([-3.14, 0.,  3.14])  # On impose les valeurs de la légende en abscisse
            i += 1

        ax[0].title.set_text(f'\u03BC = -\u03C0/2')
        ax[1].title.set_text(f'\u03BC = 0')
        ax[2].title.set_text(f'\u03BC = \u03C0/2')

        ax[0].locator_params(axis="y", nbins=4)
        # On efface les axes des ordonnées
        ax[1].get_yaxis().set_visible(False)
        ax[2].get_yaxis().set_visible(False)

        fig.suptitle(f'Von-Mises  (\u03BA = {self.kappa})')
        plt.show()
        if save:
            fig.savefig('images/describe_mu.png')

    def describe_kappa(self, save=False):
        """ This function plots the density of a von-Mises distribution for different values of kappa """

        kappa = [0, 0.1, 0.5, 1.5, 4, 20]
        fig, ax = plt.subplots(2, 3, figsize=(13, 7))

        i = 1
        x = np.linspace(-np.pi, np.pi, 300)

        ax[0, 0].plot(x, len(x) * [1 / (2 * np.pi)], 'r-', lw=1)  # On trace
        ax[0, 0].set_ylim([0., None])  # On impose que l'axe des ordonnées commence par la valeur 0.
        ax[0, 0].set_xticks([-3.14, 0., 3.14])  # On impose les valeurs de la légende en abscisse
        ax[0, 0].locator_params(axis="y", nbins=4)
        ax[0, 0].title.set_text(f'\u03BA = {0}')

        for val in kappa:
            if val != 0:
                ax[i//3, i % 3].plot(x, vonmises.pdf(x, val, loc=0.), 'r-', lw=1)  # On trace
                ax[i//3, i % 3].set_ylim([0., None])  # On impose que l'axe des ordonnées commence par la valeur 0.
                ax[i//3, i % 3].set_xticks([-3.14, 0.,  3.14])  # On impose les valeurs de la légende en abscisse
                ax[i//3, i % 3].locator_params(axis="y", nbins=4)
                ax[i//3, i % 3].title.set_text(f'\u03BA = {val}')
                i += 1

        fig.suptitle('Von-Mises (\u03BC = 0)')
        fig.tight_layout(pad=1.0)
        plt.show()
        if save:
            fig.savefig('images/describe_kappa.png')

    def simulate(self, n=1):
        raise NotImplementedError("The method should be defined in each child class.")
        pass

    def hist(self, save=False):
        """
        Generates and prints the graph of the observations made under the simulation of our model.
        User needs to run {self}.simulation(n) method first.
        :return None:
        """

        if self.results is None:
            logging.warning("No simulation has been launched yet: computing ...")
            self.simulate(n=n)

        plt.gcf().clear()
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_axes([0.065, 0.1, 0.75, 0.8])

        ax.hist(self.results, bins=200, density=True, color='grey')
        x = np.linspace(-np.pi, np.pi, 100)
        ax.plot(x, vonmises.pdf(x, self.kappa, loc=self.mu), 'r-', lw=1, label='theoretical')

        ax.text(3.85, 0.15,
                f'proposal: {self.proposal}\n\n'
                f'\u03BC:            {self.mu:.3f}\n'
                f'\u03BA:            {self.kappa:.3f}\n'
                f'n:       {self.number_observations:.1e}',
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
        if self.proposal == 'cauchy':
            self.results = von_mises_cauchy(mu=self.mu, kappa=self.kappa, n=n)
        elif self.proposal == 'uniform':
            self.results = von_mises_unif(mu=self.mu, kappa=self.kappa, n=n)
        else:
            raise NotImplementedError('Wrong proposal.')
        return self.results

    def describe_simulation(self, save=False):
        """ This function plots the simulation of a von-Mises distribution for different values of n """

        n = [10_000, 100_000, 1_000_000]

        fig, ax = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
        i = 0
        x = np.linspace(-np.pi, np.pi, 300)
        for val in n:
            self.simulate(n=val)
            ax[i].hist(self.results, bins=200, density=True, color='grey')
            ax[i].plot(x, vonmises.pdf(x, self.kappa, loc=self.mu), 'r-', lw=1, label='theoretical')
            ax[i].plot([-np.pi, np.pi], 2 * [vonmises.pdf(x, self.kappa, loc=self.mu).max()], 'b-', lw=1, label='proposal')
            ax[i].set_ylim([0., None])  # On impose que l'axe des ordonnées commence par la valeur 0.
            ax[i].set_xticks([-3.14, 0.,  3.14])  # On impose les valeurs de la légende en abscisse
            ax[i].title.set_text(f'n = {val:.1e}')
            i += 1

        ax[0].locator_params(axis="y", nbins=4)
        # On efface les axes des ordonnées
        ax[1].get_yaxis().set_visible(False)
        ax[2].get_yaxis().set_visible(False)

        fig.suptitle(f'Von-Mises  (\u03BC = {self.mu},  \u03BA = {self.kappa})')
        plt.legend(prop={'size': 8})
        plt.show()
        if save:
            fig.savefig('images/describe_simulation.png')

    def acceptance_rate_simulation(self, save=True):
        """ This function plots the acceptance rate of the rejection test for different values of kappa
        given a proposal distribution """

        n = 1_000_000
        kappa = np.linspace(0, 50, 300)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(kappa, [von_mises_cauchy_acceptance(kappa=val, n=n) for val in kappa], 'b-', lw=1, label='cauchy')
        ax.plot(kappa, [von_mises_unif_acceptance(kappa=val, n=n) for val in kappa], 'r-', lw=1, label='uniforme')
        ax.set_yticks([0,  25, 50, 75, 100])

        ax.grid(True, linewidth=0.5, color='grey', linestyle='-')
        ax.set_xlabel("\u03BA")
        ax.set_ylabel("Acceptance rate")
        ax.set_title("Acceptance rate : " + self.proposal)
        plt.legend()
        plt.show()
        if save:
            fig.savefig('images/acceptance_rate.png')


class VonMisesRWHM(ProjectModel):
    """
    Random Walk Hastings-Metropolis (RWHM) simulation for Von Mises distribution.
    """

    @staticmethod
    def proposal_step(proposal_RWHM='gaussian', sig=2.):
        """
        Computes the step of the random walk.

        :param str proposal_RWHM: proposed function
        :param float sig: standard deviation (gaussian) or size of the uniform distribution wished for our random walk
        proposal
        :return float: the random walk step
        """
        if proposal_RWHM == 'gaussian':
            return sig * normal(n=1)
        elif proposal_RWHM == 'uniform':
            return sig * uniform_1(n=1)
        else:
            raise NotImplementedError('Wrong proposal')

    def iter(self, x):
        """
    Does an iteration of the RWHM simulation.
        :param float x: initial value
        :return float: new value of the Markov chain
        """
        proposal_step = self.proposal_step(self.proposal_RWHM, self.sig)
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

        self.acceptance_rate = self.n_accept / n
        print(f'Acceptance rate: {self.acceptance_rate} %.')  # should be calibrated between 25% and 40%;

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


if __name__ == '__main__':
    # Parameters common to every model:
    mu = 0.
    kappa = 3.
    n = 1_000_000
    proposal = 'cauchy'
    save = True

    """ DESCRIBE DENSITY """
    model = VonMisesAcceptReject()
    model.describe_mu(save=save)  # plots the density for different values of mu
    model.describe_kappa(save=save)  # plots the density for different values of kappa
    model.describe_simulation(save=save)  # plots the simulation for different values of n
    model.acceptance_rate_simulation(save=save)  # plots the acceptance rate against kappa

    """ SIMULATE """
    model = VonMisesAcceptReject(mu=mu, kappa=kappa, proposal=proposal)
    model.simulate(n=n)  # generates n observations under the Accept-Reject simulation;
    model.hist(save=save)  # generates the histogram of the above observations;

    """ RANDOM WALK HASTINGS-METROPOLIS """
    # Parameters exclusive to RWHM:
    x_init = 0
    proposal_RWHM = 'gaussian'  # 'gaussian' or 'uniform'
    sig = 5.5

    model = VonMisesRWHM(mu=mu, kappa=kappa, x_init=x_init, proposal=proposal, proposal_RWHM=proposal_RWHM, sig=sig)
    # model.simulate(n=n)  # generates n observations under the Random Walk Hastings-Metropolis simulation;
    # model.hist()  # generates the histogram of the above observations;

    """ SANITY CHECK FOR RWHM """
    # model.graph_autocorrelation()
    # model.graph_chain(n_points=500)
    # implementation of burn-in sanity-check doesn't have much value there since the distribution is very narrow
    # there (between [-pi,pi]), thus the simulation cannot really 'get lost'.
