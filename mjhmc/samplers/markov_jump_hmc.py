"""
This file contains the core MJHMC algorithm, as well the algorithms for several HMC variants
  including standard HMC

As there is a significant amount of logic common to all algorithms, all of the different variants
  are implemented as classes that inherit from a common base class.
"""
import numpy as np
from mjhmc.misc.utils import overrides, min_idx, draw_from
from mjhmc.misc.distributions import Distribution
from .hmc_state import HMCState

#pylint: disable=too-many-instance-attributes
#pylint: disable=too-many-arguments

class HMCBase(object):
    """
    The base class for all HMC samplers in this file.
    Not a useful sampler in of itself but provides a useful structure
      and serves as a control
    """


    def __init__(self, Xinit=None, E=None, dEdX=None,
                 epsilon=1e-5, alpha=0.2, beta=None,
                 num_leapfrog_steps=5, distribution=None):
        """ Construct and return a new HMCBase instance

        :param Xinit: Initial configuration for position variables. Of shape (n_dims, n_batch)
        :param E: function: R^{n_dims x n_batch} -> R^{n_batch}.
          Specifies the energy of the current configuration
        :param dEdX: function: R^{n_dims x n_batch} -> R^{n_batch}
          Specifies the energy gradient of the current configuration
        :param distribution: Optional. An instance of mjhmc.misc.distributions.Distribution
          Specifies E and dEdX in place of explicit keyword arguments
        :param epsilon: step length for leapfrog integrator
        :param alpha: specifies momentum corruption rate in terms of fraction
          of momentum corrupted per sample step
        :param beta: specifies momentum corruption rate
        :param num_leapfrog_steps: number of leapfrog integration steps per application
          of L operator
        :returns: a new instance
        :rtype: HMCBase
        """
        # do not execute this block if I am an instance of MarkovJumpHMC
        if not isinstance(self, MarkovJumpHMC):
            if isinstance(distribution, Distribution):
                distribution.mjhmc = False
                distribution.reset()
                self.ndims = distribution.Xinit.shape[0]
                self.nbatch = distribution.Xinit.shape[1]
                self.energy_func = distribution.E
                self.grad_func = distribution.dEdX
                self.state = HMCState(distribution.Xinit.copy(), self)
                self.distribution = distribution
            else:
                assert Xinit is not None
                assert E is not None
                assert dEdX is not None
                self.ndims = Xinit.shape[0]
                self.nbatch = Xinit.shape[1]
                self.energy_func = E
                self.grad_func = dEdX
                self.state = HMCState(Xinit.copy(), self)

        self.num_leapfrog_steps = num_leapfrog_steps
        self.epsilon = epsilon
        self.beta = beta or alpha**(1./(self.epsilon*self.num_leapfrog_steps))

        self.original_epsilon = epsilon
        self.original_l = self.num_leapfrog_steps



        self.n_burn_in = 500

        # these settings for the base class only
        self.p_flip = 0.5
        self.p_r = 1

        # total operator counts. counted per particle
        self.l_count = 0
        self.f_count = 0
        # this one is necessary since we're not always flipping the momentum
        self.fl_count = 0
        self.r_count = 0

        # only approximate!! lower bound
        self.grad_per_sample_step = self.num_leapfrog_steps



    # to deprecate
    def E(self, X):
        """compute energy function at X"""
        E = self.energy_func(X).reshape((1,-1))
        return E

    # to deprecate
    def dEdX(self, X):
        """compute energy function gradient at X"""
        dEdX = self.grad_func(X)
        return dEdX

    def leap_prob(self, Z1, Z2):
        """
        Metropolis-Hastings Probability of transitioning from state Z1 to
        state Z2.
        """
        Ediff = Z1.H() - Z2.H()
        p_acc = np.ones((1, Ediff.shape[1]))
        p_acc[Ediff < 0] = np.exp(Ediff[Ediff < 0])
        return p_acc

    def sampling_iteration(self):
        """Perform a single sampling step
        """
        # FL operator
        proposed_state = self.state.copy().L().F()

        # Metropolis-Hasting acceptance probabilities
        p_acc = self.leap_prob(self.state, proposed_state)
        # accepted states
        fl_idx = np.arange(self.nbatch).reshape(1, self.nbatch)[np.random.rand(self.nbatch) < p_acc]
        #update accepted FL transitions
        self.state.update(fl_idx, proposed_state)

        # flip momentum with prob p_flip (.5 for control)
        # crank p_flip up to 1 to recover standard HMC
        p_half = self.p_flip * np.ones((1, self.nbatch))
        flip_idx = np.arange(self.nbatch).reshape(1, self.nbatch)[np.random.rand(self.nbatch) < p_half]

        curr_state = self.state.copy().F()
        self.state.update(flip_idx, curr_state)

        # do it particle wise
        if np.random.random() < self.p_r:
            # corrupt the momentum
            self.r_count += self.nbatch
            self.state.R()

        FL_idx = set(fl_idx)
        F_idx = set(flip_idx)

        self.l_count += len(FL_idx & F_idx)
        self.f_count += len(F_idx - FL_idx)
        self.fl_count += len(FL_idx - F_idx)

    def sample(self, n_samples=1000):
        """
        Draws nsamples, returns them all
        """
        # to do: unroll samples
        samples = []
        for _ in xrange(n_samples):
            self.sampling_iteration()
            samples.append(self.state.copy().X)
        return np.concatenate(samples, axis=1)

    def burn_in(self):
        """Runs the sample for a number of burn in sampling iterations
        """
        for _ in xrange(self.n_burn_in):
            self.sampling_iteration()


class HMC(HMCBase):
    """Implements standard HMC
    """

    def __init__(self, *args, **kwargs):
        super(HMC, self).__init__(*args, **kwargs)
        self.p_flip = 1

class ControlHMC(HMCBase):
    """Standard HMC but randomize all of the momentum some of the time
    """

    def __init__(self, *args, **kwargs):
        super(ControlHMC, self).__init__(*args, **kwargs)
        self.p_flip = 1
        self.p_r = - np.log(1 - self.beta) * 0.5
        # tells hmc state to randomize all of the momentum when R is called
        self.beta = 1


class ContinuousTimeHMC(HMCBase):
    """Base class for all markov jump HMC samplers
    """

    def __init__(self, *args, **kwargs):
        """ Initalizer method for continuous-time samplers

        :param resample: boolean flag whether to resample or not. ALWAYS set to true unless you
           have a specific reason not to. Produced samples will be biased if resample is false
        :returns: the constructed instance
        :rtype: ContinuousTimeHMC
        """
        self.resample = kwargs.pop('resample', True)
        distribution = kwargs.get('distribution')
        super(ContinuousTimeHMC, self).__init__(*args, **kwargs)
        # transformation from discrete beta to insure matching autocorrelation
        # maybe assert that beta is less than 1 if necessary
        # corrupt all of the momentum with some fixed probability
        self.p_r = - np.log(1 - self.beta) * 0.5
        # tells hmc state to randomize all of the momentum when R is called
        self.beta = 1

        if isinstance(distribution, Distribution):
            distribution.mjhmc = True
            if not distribution.generation_instance:
                distribution.reset()
            self.ndims = distribution.Xinit.shape[0]
            self.nbatch = distribution.Xinit.shape[1]
            self.energy_func = distribution.E
            self.grad_func = distribution.dEdX
            self.state = HMCState(distribution.Xinit.copy(), self)
            self.distribution = distribution
        else:
            raise NotImplementedError(
                ("Unfortunately, you must define your distribution by"
                 " subclassing mjhmc.misc.Distribution."
                 "This is due to subtle issues having to do with generating"
                 " a fair initialization for"
                 "the embedded Markov Chain. See the docs in mjhmc.misc.Distribution."
                ))

        # the last dwelling times
        self.dwelling_times = np.zeros(self.nbatch)




    @overrides(HMCBase)
    def sampling_iteration(self):
        """Perform a single sampling step
        """
        # F operator
        f_state = self.state.copy().F()

        # FL operator
        fl_state = self.state.copy().L().F()

        # rates
        fl_rates = self.transition_rates(self.state, fl_state)
        f_rates = np.ones((1, self.nbatch))
        r_rates = self.p_r * np.ones((1, self.nbatch))

        # draws from exponential distributions
        fl_draws = draw_from(fl_rates[0])
        f_draws = draw_from(f_rates[0])
        r_draws = draw_from(r_rates[0])

        # choose min for each particle
        f_idx, fl_idx, r_idx = min_idx([f_draws, fl_draws, r_draws])

        # record dwelling times
        self.dwelling_times = np.amin(
            np.concatenate((fl_draws, f_draws, r_draws)), axis=0)

        # update accepted FL transitions
        self.state.update(fl_idx, fl_state)

        # update accepted F transitions
        self.state.update(f_idx, f_state)

        # corrupt the momentum and update accepted R transition
        # inefficiently corrupts momentum for all state then selects a subset
        R_state = self.state.copy().R()
        self.state.update(r_idx, R_state)

        self.fl_count  += len(fl_idx)
        self.f_count += len(f_idx)
        self.r_count += len(r_idx)

    @overrides(HMCBase)
    def sample(self, n_samples=1000):
        """ Runs sampler and returns a list of n_samples (resampled to be fair)

        :param n_samples: number of samples_k to generated
        :rtype: array
        """
        if self.resample:
            samples_k = []
            dwell_t_k = []
            resamples = np.zeros((self.ndims, n_samples * self.nbatch))

            self.sampling_iteration()
            samples_k.append(self.state.copy().X)
            for _ in xrange(n_samples):
                dwell_t_k.append(self.dwelling_times.copy())
                self.sampling_iteration()
                samples_k.append(self.state.copy().X)

            dwell_t = np.concatenate(dwell_t_k)
            samples = np.concatenate(samples_k[:-1], axis=1)
            total_t = np.sum(dwell_t)
            cumul_t = np.cumsum(dwell_t)
            # pretty sure there's a way to do the whole batch at once
            for idx, rand_val in enumerate(np.sort(np.random.random(n_samples * self.nbatch)) * total_t):
                sample_idx = np.where(cumul_t > rand_val)[0][0]
                resamples[:, idx] = samples[:, sample_idx]
            return resamples
        else:
            samples = []
            for _ in xrange(n_samples):
                self.sampling_iteration()
                samples.append(self.state.copy().X)
            return np.concatenate(samples, axis=1)


    def transition_rates(self, Z1, Z2):
        """
        transition rate from Z1 to Z2
        rate = [p(Z1)/p(Z2)]^1/2
        """
        Ediff = Z1.H() - Z2.H()
        return np.exp(Ediff)**.5


class MarkovJumpHMC(ContinuousTimeHMC):
    """This class implements Markov Jump HMC as described in http://arxiv.org/abs/1509.03808
    """

    @overrides(ContinuousTimeHMC)
    def sampling_iteration(self):
        # states
        f_state = self.state.copy().F()
        l_state = self.state.copy().L()
        # aka L^-1 state
        flf_state = self.state.copy().FLF()
        r_state = self.state.copy().R()


        try:
            # rates
            l_rates = self.transition_rates(self.state, l_state)
            flf_rates = self.transition_rates(self.state, flf_state)
            f_rates = flf_rates - np.min((flf_rates, l_rates), axis=0)
            r_rates = self.p_r * np.ones((1, self.nbatch))

            # draws from exponential distributions
            l_draws = draw_from(l_rates[0])
            f_draws = draw_from(f_rates[0])
            r_draws = draw_from(r_rates[0])
        # infinite rate due to taking too large of a step
        except ValueError:
            # take smaller steps, but go the same overall distance
            self.epsilon *= 0.5
            self.num_leapfrog_steps *= 2

            depth = np.log(self.original_epsilon / self.epsilon) / np.log(2)
            print("Ecountered infinite rate, doubling back. Depth: {}".format(depth))
            # try again
            self.state.reset_flf_cache()
            self.sampling_iteration()
            # restore the old guys
            self.epsilon *= 2
            self.num_leapfrog_steps = int(self.num_leapfrog_steps / 2)
            return

        # choose min for each particle
        l_idx, f_idx, r_idx = min_idx([l_draws, f_draws, r_draws])

        # record dwelling times
        self.dwelling_times = np.amin(
            np.concatenate((l_draws, f_draws, r_draws)), axis=0)

        # cache current state as FLF state for next L transition
        self.state.cache_flf_state(l_idx, self.state)
        # cache FL as FLF state for for particles that made transition to F
        # self.state.cache_flf_state(f_idx, l_state.F())

        # update accepted proposed states
        self.state.update(l_idx, l_state)
        self.state.update(f_idx, f_state)
        self.state.update(r_idx, r_state)

        # clear flf cache for particles that transition to R, F
        self.state.clear_flf_cache(r_idx)
        self.state.clear_flf_cache(f_idx)


        self.l_count += len(l_idx)
        self.f_count += len(f_idx)
        self.r_count += len(r_idx)
