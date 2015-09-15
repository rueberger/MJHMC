import unittest
from search.objective import fit
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
        t = np.arange(self.t_max)
        x_1 = np.exp(-t)
        x_2 = np.exp(-2 * t)
        x_3 = np.exp(-3 * t)
        self.assertTrue(
            fit(t, x_1) > fit(t, x_2),
            msg="Monotonicity failure"
        )
        self.assertTrue(
            fit(t, x_2) > fit(t, x_3),
            msg="Monotonicity failure"
        )
        self.assertTrue(
            fit(t, x_1) > fit(t, x_3),
            msg="Monotonicity failure"
        )

    def TestNoise(self):
        """
        Monotonicity plus varying amount of noise
        """
        t = np.arange(self.t_max)
        x_1 = np.exp(-t) + np.random.randn(self.t_max)
        x_2 = np.exp(-2 * t) + np.random.randn(self.t_max)
        x_3 = np.exp(-3 * t) + np.random.randn(self.t_max)
        self.assertTrue(
            fit(t, x_1) > fit(t, x_2),
            msg="Noisy monotonicity failure"
        )
        self.assertTrue(
            fit(t, x_2) > fit(t, x_3),
            msg="Noisy monotonicity failure"
        )
        self.assertTrue(
            fit(t, x_1) > fit(t, x_3),
            msg="Noisy monotonicity failure"
        )

    def TestErroneous(self):
        """
        Test weird shapes, like flat lines other things
        """
        t = np.arange(self.t_max)
        x_1 = np.exp(-t) + np.random.randn(self.t_max)
        x_flat = np.ones(self.t_max)
        self.assertTrue(
            fit(t, x_flat) > fit(t, x_1),
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
            fit(t, x_1) > fit(t, x_2),
            msg="Long trial monotonicity failure"
        )
        self.assertTrue(
            fit(t, x_1_noise) > fit(t, x_2_noise),
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
            fit(t, x_nan) > fit(t, x_1),
            msg="Nan fit failure"
        )


    def TestEndNoise(self):
        """
        Test that a bunch of noise at the end doesn't bork everything
        """
        t = np.arange(self.t_max)
        x_1 = np.exp(-t)
        x_rand_end = np.ones(self.t_max)
        x_rand_end[-3:] = 10 * np.random.randn(3)
        x_pos_end = np.ones(self.t_max)
        x_pos_end[-3:] *= 100
        x_neg_end = np.ones(self.t_max)
        x_neg_end[-3:] *= -100
        self.assertTrue(
            fit(t, x_rand_end) > fit(t, x_1),
            msg="Random end noise does better than exponential"
        )
        self.assertTrue(
            fit(t, x_pos_end) > fit(t, x_1),
            msg="Positive end noise wins"
        )
        self.assertTrue(
            fit(t, x_neg_end) > fit(t, x_1),
            msg="Negative end noise wins"
        )
