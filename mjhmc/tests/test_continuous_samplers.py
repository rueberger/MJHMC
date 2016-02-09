import unittest
from mjhmc.samplers.markov_jump_hmc import ContinuousTimeHMC, HMCBase, MarkovJumpHMC, HMC, ControlHMC
from mjhmc.misc.distributions import TestGaussian, Gaussian
import numpy as np
from mjhmc.misc.utils import overrides

n_seed = 1
eps = .05

class TestControl(unittest.TestCase):
    """
    Serves as the base class since unittests messes up inheritance
    """

    def setUp(self):
        np.random.seed(n_seed)
        self.sampler_to_test = HMCBase

    def test_1d_gaussian(self):
        """
        Checks to see that {} can sample from a 1d gaussian
        """.format(self.sampler_to_test.__name__)
        n_samples = 10000
        gaussian_1d = TestGaussian(ndims=1)
        sampler = self.sampler_to_test(
            gaussian_1d.Xinit,
            gaussian_1d.E,
            gaussian_1d.dEdX
        )
        sampler.burn_in()
        samples = sampler.sample(n_samples)
        mean = np.mean(samples)
        std = np.std(samples)
        self.assertTrue(np.abs(mean) < eps,
                        msg='mean: {} is not within tolerance for {}'.format(
                            mean,
                            self.sampler_to_test.__name__))
        self.assertTrue(np.abs(std - 1) < eps,
                        msg='std: {} is not within tolerance for {}'.format(
                            std,
                            self.sampler_to_test.__name__))

    def test_ill_conditioned_gaussian(self):
        """
        Checks to see that {} can sample from an ill-conditioned gaussian
        """.format(self.sampler_to_test.__name__)
        n_samples = 100000
        ic_gaussian_2d = Gaussian(ndims=2, log_conditioning=1)
        target_cov = np.linalg.inv(ic_gaussian_2d.J)
        sampler = self.sampler_to_test(
            ic_gaussian_2d.Xinit,
            ic_gaussian_2d.E,
            ic_gaussian_2d.dEdX
        )
        sampler.burn_in()
        samples = sampler.sample(n_samples)
        sample_cov = np.cov(samples)
        self.assertTrue(self.approx_equal(sample_cov, target_cov),
                        msg=(" samples covariance: \n {} is not within tolerance to "
                             "target cov: \n {}. \n I am {}").format(
                                 sample_cov,
                                 target_cov,
                                 self.sampler_to_test.__name__))

    def approx_equal(self, arr1, arr2):
        return np.linalg.norm(arr1 - arr2) < eps


    def test_hyperparameter_setting(self):
        """
        Checks to see that hyperparameters are properly set
        """
        beta = np.random.random()
        epsilon = np.random.random() * 5
        num_leapfrop_steps = np.random.randint(10)
        gauss = Gaussian()
        sampler = self.sampler_to_test(
            distribution=gauss,
            beta=beta,
            epsilon=epsilon,
            num_leapfropg_steps=num_leapfrop_steps
        )
        self.assertTrue(sampler.beta == beta)
        self.assertTrue(sampler.epsilon == epsilon)
        self.assertTrue(sampler.num_leapfropg_steps == num_leapfrop_steps)

class TestHMC(TestControl):

    @overrides(TestControl)
    def setUp(self):
        np.random.seed(n_seed)
        self.sampler_to_test = HMC

class TestControlHMC(TestControl):

    @overrides(TestControl)
    def setUp(self):
        np.random.seed(n_seed)
        self.sampler_to_test = ControlHMC


# class TestContinuousHMC(TestControl):

#     @overrides(TestControl)
#     def setUp(self):
#         np.random.seed(n_seed)
#         self.sampler_to_test = ContinuousTimeHMC


class TestMJHMC(TestControl):

    @overrides(TestControl)
    def setUp(self):
        np.random.seed(n_seed)
        self.sampler_to_test = MarkovJumpHMC
