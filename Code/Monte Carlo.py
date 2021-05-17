import logging
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import vonmises
from scipy.optimize import minimize
from statsmodels.graphics.tsaplots import plot_acf
from tools import *


class ProjectModel:
    """
    Parent class to all our project models classes, gathering common methods and constructor for every von Mises models.
    """

    def __init__(self, mu=0., kappa=1., x_init=0., proposal='cauchy', proposal_RWHM='gaussian', sig=2.):
        self.mu = mu
        self.kappa = kappa

        # Proposed distribution: 'cauchy' or 'uniform'
        self.proposal = proposal

        if self.__class__.__name__ == 'VonMisesRWHM':
            # The following attributes are built if the object is initialized with the class 'VonMisesRWHM'

            # Initialisation value of the Markov chain;
            self.x_init = x_init

            # Proposed distribution for the random walk: 'gaussian' or 'uniform'
            self.proposal_RWHM = proposal_RWHM

            # Standard deviation (gaussian) or size of the uniform distribution wished for our random walk proposal;
            self.sig = sig

            self.n_accept = 0
            self.acceptance_rate = 0

        # Number of observations the simulation will predict
        self.number_observations = None
        self.results = None

    def estimate_params_MCMC_MLE(self, n=500, m=10_000):
        """ This function estimates the parameters of a von Mises ditribution by MCMC MLE

        :param int n: the number of simulations to estimate the  parameters (mu, kappa)
        :param int m: the number of simulations to estimate the log-likelihood

        :return array (2, 2): numpy array of the parameters and the estimated std
        """

        def log_likelihood(params):
            mu, kappa = params
            res_1 = kappa * np.cos(sample - mu) - np.cos(sample)
            res_2 = np.exp(kappa * np.cos(sample_ - mu) - np.cos(sample_))
            return -res_1.mean() + np.log(res_2.mean())

        params_estimated = np.zeros((n, 2))
        model_ = VonMisesAcceptReject(mu=0., kappa=1.)

        for i in range(n):
            sample = self.simulate(n=m)
            sample_ = model_.simulate(n=m)
            params_estimated[i] = minimize(log_likelihood, np.array([0., 1.])).x

        mu_, kappa_ = params_estimated.mean(axis=0)
        mu_std, kappa_std = params_estimated.std(axis=0) * np.sqrt(n / (n - 1))

        print(
            f"Parameters:\n\u03BC = {mu_:.5f}  [{mu_ - 1.96 * mu_std / n ** 0.5:.5f}, {mu_ + 1.96 * mu_std / n ** 0.5:.5f}]")
        print(
            f"\u03BA = {kappa_:.5f}  [{kappa_ - 1.96 * kappa_std / n ** 0.5:.5f}, {kappa_ + 1.96 * kappa_std / n ** 0.5:.5f}]\n")

        return np.array([[mu_, mu_std], [kappa_, kappa_std]])

    def describe_mu(self, save=False):
        """ This function plots the density of a von Mises distribution for different values of mu """

        mu = [-np.pi / 2, 0, np.pi / 2]
        x = np.linspace(-np.pi, np.pi, 300)

        fig, ax = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
        i = 0

        for val in mu:
            ax[i].plot(x, vonmises.pdf(x, self.kappa, loc=val), 'r-', lw=1)
            ax[i].set_ylim([0., None])
            ax[i].set_xticks([-3.14, 0., 3.14])
            i += 1

        ax[0].title.set_text(f'\u03BC = -\u03C0/2')
        ax[1].title.set_text(f'\u03BC = 0')
        ax[2].title.set_text(f'\u03BC = \u03C0/2')

        ax[0].locator_params(axis="y", nbins=4)
        ax[1].get_yaxis().set_visible(False)
        ax[2].get_yaxis().set_visible(False)

        fig.suptitle(f'von Mises  (\u03BA = {self.kappa})')
        plt.show()
        if save:
            fig.savefig('Graphs/describe_mu.png')

    def describe_kappa(self, save=False):
        """ This function plots the density of a von Mises distribution for different values of kappa """

        kappa = [0, 0.1, 0.5, 1.5, 4, 20]
        x = np.linspace(-np.pi, np.pi, 300)

        fig, ax = plt.subplots(2, 3, figsize=(13, 7))
        i = 1

        ax[0, 0].plot([-np.pi, np.pi], 2 * [1 / (2 * np.pi)], 'r-', lw=1)
        ax[0, 0].set_ylim([0., None])
        ax[0, 0].set_xticks([-3.14, 0., 3.14])
        ax[0, 0].locator_params(axis="y", nbins=4)
        ax[0, 0].title.set_text(f'\u03BA = {0}')

        for val in kappa:
            if val != 0:
                ax[i // 3, i % 3].plot(x, vonmises.pdf(x, val, loc=0.), 'r-', lw=1)
                ax[i // 3, i % 3].set_ylim([0., None])
                ax[i // 3, i % 3].set_xticks([-3.14, 0., 3.14])
                ax[i // 3, i % 3].locator_params(axis="y", nbins=4)
                ax[i // 3, i % 3].title.set_text(f'\u03BA = {val}')
                i += 1

        fig.suptitle('von Mises (\u03BC = 0)')
        fig.tight_layout(pad=1.0)
        plt.show()
        if save:
            fig.savefig('Graphs/describe_kappa.png')

    def simulate(self, n=1):
        raise NotImplementedError("The method should be defined in each child class.")
        pass

    def hist(self):
        """
        Generates and plots the graph of the observations made under the simulation of our model.
        """

        if self.results is None:
            logging.warning("No simulation has been launched yet: computing ...")
            self.simulate(n=1)

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

        ax.set_title(f'von Mises simulation : {self.__class__.__name__}.')
        plt.legend()
        plt.show()


class VonMisesAcceptReject(ProjectModel):
    """
    Accept-Reject simulation for von Mises distribution.
    """

    def simulate(self, n=1):
        """
        Generates n simulations under our von Mises Accept-Reject simulation.

        :param int n: number of simulations
        :return array: list of simulations
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
        """ This function plots the simulation of a von Mises distribution for different values of n """

        n = [10_000, 100_000, 1_000_000]
        x = np.linspace(-np.pi, np.pi, 300)

        fig, ax = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
        i = 0

        for val in n:
            self.simulate(n=val)
            ax[i].hist(self.results, bins=200, density=True, color='grey')
            ax[i].plot(x, vonmises.pdf(x, self.kappa, loc=self.mu), 'r-', lw=1, label='theoretical')

            if self.proposal == 'uniform':
                ax[i].plot([-np.pi, np.pi], 2 * [vonmises.pdf(x, self.kappa, loc=self.mu).max()], 'b-', lw=1,
                           label='proposal')

            elif self.proposal == 'cauchy':
                y, x = np.histogram(wrapped_cauchy(mu=self.mu, kappa=self.kappa, n=10_000_000),
                                    bins=np.linspace(-np.pi, np.pi, 1_000), density=True)
                y_, _ = np.histogram(von_mises_cauchy(mu=self.mu, kappa=self.kappa, n=10_000_000),
                                     bins=np.linspace(-np.pi, np.pi, 1_000), density=True)
                ax[i].plot((x[:-1] + x[1:]) / 2, (y_ / y).max() * gaussian_filter1d(y, sigma=10), 'b-', lw=1,
                           label='proposal')

            ax[i].set_ylim([0., None])
            ax[i].set_xticks([-3.14, 0., 3.14])
            ax[i].title.set_text(f'n = {val:.1e}')
            i += 1

        ax[0].locator_params(axis="y", nbins=4)
        ax[1].get_yaxis().set_visible(False)
        ax[2].get_yaxis().set_visible(False)

        fig.suptitle(f'von Mises  (\u03BC = {self.mu:.3f},  \u03BA = {self.kappa:.3f},  proposal = {self.proposal})')
        plt.legend(prop={'size': 8})
        plt.show()
        if save:
            fig.savefig('Graphs/describe_simulation_' + self.proposal + '.png')

    def acceptance_rate_simulation(self, save=True):
        """ This function plots the acceptance rate of the rejection test for different values of kappa
        for each proposal distribution """

        n = 1_000_000
        kappa = np.linspace(0, 50, 300)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(kappa, [von_mises_cauchy_acceptance(kappa=val, n=n) for val in kappa], 'b-', lw=1,
                label='wrapped cauchy')
        ax.plot(kappa, [von_mises_unif_acceptance(kappa=val, n=n) for val in kappa], 'r-', lw=1, label='uniform')
        ax.set_yticks([0, 25, 50, 75, 100])

        ax.grid(True, linewidth=0.5, color='grey', linestyle='-')
        ax.set_xlabel("\u03BA")
        ax.set_ylabel("Acceptance rate")
        ax.set_title("Acceptance rate")
        plt.legend()
        plt.show()
        if save:
            fig.savefig('Graphs/acceptance_rate.png')


class VonMisesRWHM(ProjectModel):
    """
    Random Walk Hastings-Metropolis (RWHM) simulation for von Mises distribution.
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

    def simulate(self, n=100_000, print_acceptance_rates=True):
        """
        Generates n simulations under our von Mises Random Walk Hastings-Metropolis simulation.
        :param int n: number of simulations
        :return list: list of simulations
        """

        self.number_observations = n
        self.n_accept = 0

        x = np.empty(n)
        x[0] = self.x_init
        for i in range(1, n):  # we realize (n-1) iterations of the Markov Chain
            x[i] = self.iter(x[i - 1])

        self.acceptance_rate = self.n_accept / n
        if print_acceptance_rates == True: print(
            f'Acceptance rate: {self.acceptance_rate * 100} %.')  # should be calibrated between 25% and 40%;

        self.results = x
        return self.results

    def describe_simulation(self, save=False):
        """ This function plots the simulation of a von Mises distribution for different values of n """

        n = [10_000, 100_000, 1_000_000]
        x = np.linspace(-np.pi, np.pi, 300)

        fig, ax = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
        i = 0

        for val in n:
            self.simulate(n=val)
            ax[i].hist(self.results, bins=200, density=True, color='grey')
            ax[i].plot(x, vonmises.pdf(x, self.kappa, loc=self.mu), 'r-', lw=1, label='theoretical')

            ax[i].set_ylim([0., None])
            ax[i].set_xticks([-3.14, 0., 3.14])
            ax[i].title.set_text(f'n = {val:.1e}')
            i += 1

        ax[0].locator_params(axis="y", nbins=4)
        ax[1].get_yaxis().set_visible(False)
        ax[2].get_yaxis().set_visible(False)

        fig.suptitle(
            f'von Mises  (\u03BC = {self.mu:.3f},  \u03BA = {self.kappa:.3f},  proposal_RWHM = {self.proposal_RWHM})')
        plt.legend(prop={'size': 8})
        plt.show()
        if save:
            fig.savefig('Graphs/describe_simulation_RWHM_' + self.proposal_RWHM + '.png')

    def fit(self, len_batch=1000, num_iter=1000):
        """
        Chooses an optimal value for sigma, the standard deviation of our random walk. Keep in mind when choosing
        parameters that the number of simulation that will be computed is len_batch * num_iter.
        :param int len_batch: number of simulations for each Von Mises estimation.
        :param int num_iter: number of sigma that we want to try.
        :return NoneType: None. Prints the advised value for sigma.
        """
        old_sig = self.sig  # Stocks the value of sigma initially entered by user.
        step = 10 / num_iter
        sigma_proposals = [step * i for i in range(1, num_iter)]
        for i in range(num_iter):
            sig = sigma_proposals[i]
            self.sig = sig
            self.simulate(len_batch, print_acceptance_rates=False)
            if 0.3 < self.acceptance_rate < 0.4:
                print(f'Recommended value for sigma : {sig}. Acceptance rate around {self.acceptance_rate}.')
                print(i)
                return None
        self.sig = old_sig  # In order not to modify the sigma value wished by the user.
        print("Couldn't find a sigma that satisfies the acceptance rate criteria.")

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
    mu = 2
    kappa = 14
    n = 1_0000
    proposal = 'cauchy'
    save = False

    """ DESCRIBE """
    # model = VonMisesAcceptReject(mu=mu, kappa=kappa) model.describe_mu(save=save)  # plots the density for
    # different values of mu model.describe_kappa(save=save)  # plots the density for different values of kappa
    # model.describe_simulation(save=save)  # plots the simulation for different values of n VonMisesAcceptReject(
    # mu=mu, kappa=kappa, proposal='uniform').describe_simulation(save=save)  # plots the simulation for different
    # values of n model.estimate_params_MCMC_MLE()  # estimates the parameters of the model by MCMC MLE
    # model.acceptance_rate_simulation(save=save)  # plots the acceptance rate against kappa

    """ SIMULATE """
    # model = VonMisesAcceptReject(mu=mu, kappa=kappa, proposal=proposal)
    # model.simulate(n=n)  # generates n observations under the Accept-Reject simulation
    # model.hist()  # generates the histogram of the above observations

    """ RANDOM WALK HASTINGS-METROPOLIS """
    # Parameters exclusive to RWHM:
    x_init = 0
    proposal_RWHM = 'gaussian'  # 'gaussian' or 'uniform'
    sig = 5.5

    model_RWHM = VonMisesRWHM(mu=mu, kappa=kappa, x_init=x_init, proposal=proposal, proposal_RWHM=proposal_RWHM,
                              sig=sig)
    model_RWHM.fit(len_batch=100, num_iter=1_000)
    # model_RWHM.simulate(n=n)  # generates n observations under the Random Walk Hastings-Metropolis simulation;
    # model_RWHM.hist()  # generates the histogram of the above observations;
    # model_RWHM.describe_simulation(save=save)  # plots the simulation for different values of n

    """ SANITY CHECK FOR RWHM """
    # model_RWHM.graph_autocorrelation()
    # model_RWHM.graph_chain(n_points=500)
    # implementation of burn-in sanity-check doesn't have much value there since the distribution is very narrow
    # there (between [-pi,pi]), thus the simulation cannot really 'get lost'.
