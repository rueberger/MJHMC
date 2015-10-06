import unittest
from mjhmc.search.objective import fit
import numpy as np

n_seed = 1

class TestObjetive(unittest.TestCase):
    """
    Tests the search objective function
    objective function tested is the parameter for an exponential decay fit
    """

    def setUp(self):
        np.random.seed(n_seed)
        self.t_max = 1000

    def TestMonotonicity(self):
        """
        Tests to see that objective function is monotonic
        """
        t = np.arange(self.t_max) / float(self.t_max)
        x_1 = np.exp(-t)
        x_2 = np.exp(-2 * t)
        x_3 = np.exp(-3 * t)
        self.assertTrue(
            fit(t, x_1)[0] > fit(t, x_2)[0],
            msg="Monotonicity failure"
        )
        self.assertTrue(
            fit(t, x_2)[0] > fit(t, x_3)[0],
            msg="Monotonicity failure"
        )
        self.assertTrue(
            fit(t, x_1)[0] > fit(t, x_3)[0],
            msg="Monotonicity failure"
        )

    def TestNoise(self):
        """
        Monotonicity plus varying amount of noise
        """
        perc_noise = 0.01
        t = np.arange(self.t_max) / float(self.t_max)
        x_1 = np.exp(-t) + np.random.randn(self.t_max) * perc_noise
        x_2 = np.exp(-2 * t) + np.random.randn(self.t_max) * perc_noise
        x_3 = np.exp(-3 * t) + np.random.randn(self.t_max) * perc_noise
        self.assertTrue(
            fit(t, x_1)[0] > fit(t, x_2)[0],
            msg="Monotonicity failure"
        )
        self.assertTrue(
            fit(t, x_2)[0] > fit(t, x_3)[0],
            msg="Monotonicity failure"
        )
        self.assertTrue(
            fit(t, x_1)[0] > fit(t, x_3)[0],
            msg="Monotonicity failure"
        )

    def TestErroneous(self):
        """
        Test weird shapes, like flat lines other things
        """
        t = np.arange(self.t_max) / float(self.t_max)
        x_1 = np.exp(-t) + np.random.randn(self.t_max)
        x_flat = np.ones(self.t_max)
        self.assertTrue(
            fit(t, x_flat)[0] > fit(t, x_1)[0],
            msg="Constant better than exponential"
        )

    def TestLongTrial(self):
        """
        Test extra long trial to check for floating point craziness
        """
        t = np.arange(1E5)
        x_1 = np.exp(-t)
        x_2 = np.exp(-2 * t)
        x_1_noise = np.exp(-t) + 0.1 * np.random.randn(1E5)
        x_2_noise = np.exp(-2 * t) + 0.1 * np.random.randn(1E5)
        self.assertTrue(
            fit(t, x_1)[0] > fit(t, x_2)[0],
            msg="Long trial monotonicity failure"
        )
        self.assertTrue(
            fit(t, x_1_noise)[0] > fit(t, x_2_noise)[0],
            msg="Long trial noisy monotonicity failure"
        )


    def TestNan(self):
        """
        Test that nan values are higher than any non nan value
        """
        t = np.arange(1E5)
        x_1 = np.exp(-t)
        nan_idx = np.random.randint(1E5, size=5)
        x_nan = np.exp(-2 * t)
        x_nan[nan_idx] = np.nan
        self.assertTrue(
            fit(t, x_nan)[0] > fit(t, x_1)[0],
            msg="Nan fit failure"
        )


    def TestEndNoise(self):
        """
        Test that a bunch of noise at the end doesn't bork everything
        """
        t = np.arange(self.t_max) / float(self.t_max)
        x_1 = np.exp(-2 * t)
        x_rand_end = np.ones(self.t_max)
        x_rand_end[-3:] = 10 * np.random.randn(3)
        x_pos_end = np.ones(self.t_max)
        x_pos_end[-3:] *= 100
        x_neg_end = np.ones(self.t_max)
        x_neg_end[-3:] *= -100
        self.assertTrue(
            fit(t, x_rand_end)[0] > fit(t, x_1)[0],
            msg="Random end noise does better than exponential"
        )
        self.assertTrue(
            fit(t, x_pos_end)[0] > fit(t, x_1)[0],
            msg="Positive end noise wins"
        )
        self.assertTrue(
            fit(t, x_neg_end)[0] > fit(t, x_1)[0],
            msg="Negative end noise wins"
        )
