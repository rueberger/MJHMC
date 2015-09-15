import unittest
from mjhmc.samplers.generic_discrete import Gibbs, Continuous, IsingState
from itertools import product
import numpy as np
from mjhmc.misc.utils import overrides

n_seed = 1
eps = .1

class TestIsingGibbs(unittest.TestCase):
    """
    base class for tests of the generic discrete samplers
    this class tests the gibbs sampler on the ising model
    """

    def setUp(self):
        np.random.seed(n_seed)
        self.sampler_to_test = Gibbs

    @unittest.skip("Ising broken right now")
    def testConvergence(self):
        """
        tests that the sampler converges to expected distribution
        uses a random J
        """
        n_samples = 10000
        self.ndims = 2
        self.nbatch = 100
        ising = IsingState(self.ndims, self.nbatch)
        sampler = self.sampler_to_test(ising)

        emp_distr_dict = {}
        true_distr_dict = self.calculate_true_distribution(ising.J, self.ndims)
        true_distr = np.zeros(2**self.ndims)
        emp_distr = np.zeros(2**self.ndims)

        # burn in
        sampler.sample(1000)

        for x in sampler.sample(n_samples).T:
            t_x = tuple(x)
            if t_x in emp_distr_dict:
                emp_distr_dict[t_x] += 1. / (n_samples * self.nbatch)
            else:
                emp_distr_dict[t_x] = 1. / (n_samples * self.nbatch)

        for i, (x, true_p) in enumerate(true_distr_dict.iteritems()):
            true_distr[i] = true_p
            emp_distr[i] = emp_distr_dict.get(x) or 0

        self.assertTrue(
            np.linalg.norm(true_distr - emp_distr) < eps,
            msg="empirical distribution was not within tolerance to the true distribution for {}".format(
                self.sampler_to_test.__name__
            )
        )

    def calculate_true_distribution(self, J, ndims):
        """
        J : the connectivity matrix
        returns the true distribution
        """
        Z = 0
        distr = {}
        for x in product([-1, 1], repeat=ndims):
            x_arr = np.array(x)
            p = np.exp(- x_arr.dot(J).dot(x_arr.T))
            Z += p
            distr[x] = p
        for x, p in distr.iteritems():
            distr[x] = p / Z
        return distr


class TestIsingContinuous(TestIsingGibbs):

    @overrides(TestIsingGibbs)
    def setUp(self):
        np.random.seed(n_seed)
        self.sampler_to_test = Continuous


class TestIsingState(unittest.TestCase):
    """
    Tests the IsingState class
    """

    def setUp(self):
        np.random.seed(n_seed)
        self.ndims = 8
        self.nbatch = 10

    def testHamiltonian(self):
        """
        tests the hamiltonian method on a random J
        """
        ising = IsingState(self.ndims, self.nbatch)
        J = ising.J
        for x in product([-1, 1], repeat=self.ndims):
            x_arr = np.array(x)
            expected_H = x_arr.dot(J).dot(x_arr.T)
            ising.X[:,0] = x_arr
            actual_H = ising.H()[0]
            self.assertEqual(expected_H, actual_H,
                            msg="Calculated energies do not match"
            )
