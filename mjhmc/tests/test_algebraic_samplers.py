import unittest
from mjhmc.samplers.algebraic_hmc import (StateGroup, AlgebraicDiscrete,
                                    AlgebraicContinuous, AlgebraicReducedFlip,
                                    AlgebraicHMC)
import itertools
import numpy as np
from mjhmc.misc.utils import overrides

n_seed = 1
# error tolerance
EPS = .01


class TestAlgebraicDiscrete(unittest.TestCase):
    """
    Base test class for algebraic samplers

    Inheritance in unittest is funky so I'm just setting the discrete sampler
    to be the base class
    """

    def setUp(self):
        np.random.seed(n_seed)
        self.n_sampler_steps = 5000
        self.sampler_to_test = AlgebraicDiscrete

    def test_analytic_transition_matrix(self):
        """
        tests that the computed analytic matrix is within a tolerance to the empirical matrix
        uses np.arange(order / 2) as the test distribution for the energies. Tests {}
        """.format(self.sampler_to_test.__name__)
        self.max_n_energies = 5
        for order in (np.arange(2, self.max_n_energies) * 2):
            sampler = self.sampler_to_test(
                order,
                energies=np.arange(order / 2)
            )
            sampler.sample(self.n_sampler_steps, burn_in=True)
            analytic_T = sampler.calculate_true_transition_matrix(False)
            empirical_T = sampler.get_empirical_transition_matrix()
            self.assertTrue(
                self.approx_equal(analytic_T, empirical_T),
                "analytic matrix: \n {} is not within tolerance of empirical: \n {}. I am {}".format(
                    analytic_T, empirical_T, self.sampler_to_test.__name__
                )
            )

    def approx_equal(self, arr1, arr2):
        return np.linalg.norm(arr1 - arr2) < EPS

    def get_distribution(self, energies):
        """
        helper method that initializes the sampler with energies
        and returns empirical_distr, theoretical_distribution
        """
        sampler = self.sampler_to_test(
            len(energies) * 2,
            energies
        )
        sampler.sample(self.n_sampler_steps, burn_in=True)
        return sampler.get_empirical_distr(), sampler.calculate_true_distribution()

    def test_convergence(self):
        """
        test that the algebraic samplers converge to the right distribution
        """
        # 2d uniform
        self.assertTrue(
            self.approx_equal(*self.get_distribution(np.ones(2))),
            msg="did not converge to 2d uniform distribution "
        )
        # 5d uniform
        self.assertTrue(
            self.approx_equal(*self.get_distribution(np.ones(5))),
            msg="did not converge to 5d uniform distribution "
        )
        # 10d uniform
        self.assertTrue(
            self.approx_equal(*self.get_distribution(np.ones(10))),
            msg="did not converge to 10d uniform distribution "
        )
        # 5d linear
        self.assertTrue(
            self.approx_equal(*self.get_distribution(np.arange(5))),
            msg="did not converge to 5d linear distribution (np.arange(5)) "
        )
        # 5d randn
        self.assertTrue(
            self.approx_equal(*self.get_distribution(np.random.randn(5))),
            msg="did not converge to 5d gaussian random "
        )


class TestAlgebraicHMC(TestAlgebraicDiscrete):

    @overrides(TestAlgebraicDiscrete)
    def setUp(self):
        np.random.seed(n_seed)
        self.n_sampler_steps = 5000
        self.sampler_to_test = AlgebraicHMC


class TestAlgebraicContinuous(TestAlgebraicDiscrete):

    @overrides(TestAlgebraicDiscrete)
    def setUp(self):
        np.random.seed(n_seed)
        self.n_sampler_steps = 5000
        self.sampler_to_test = AlgebraicContinuous


class TestAlgebraicReducedFlip(TestAlgebraicDiscrete):

    @overrides(TestAlgebraicDiscrete)
    def setUp(self):
        np.random.seed(n_seed)
        self.n_sampler_steps = 5000
        self.sampler_to_test = AlgebraicReducedFlip


class TestStateGroup(unittest.TestCase):

    def test_full_indexing(self):
        """
        test that self.full_idx is bijective
        """
        max_order = 100
        for order in (np.arange(2, max_order) * 2):
            ladder = StateGroup(order, np.ones(order / 2))
            idx_map = {}
            for f_k, l_k in itertools.product(np.arange(2), np.arange(order / 2)):
                ladder.state = [f_k, l_k]
                idx = ladder.full_idx()
                self.assertNotIn(
                    idx,
                    idx_map,
                    "current state: {} conflicts with previous for index {}".format(
                        [f_k, l_k], idx
                    )
                )
                idx_map[idx] = [f_k, l_k]
