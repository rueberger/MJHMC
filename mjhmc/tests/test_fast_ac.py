"""
This module contains unit tests for the theano autocorrelation function
"""
import unittest
import numpy as np

from mjhmc.fast.hmc import normed_autocorrelation
from mjhmc.misc.autocor import autocorrelation, sample_to_df

from mjhmc.misc.distributions import Gaussian
from mjhmc.samplers.markov_jump_hmc import MarkovJumpHMC


class TestFastAutocorrelation(unittest.TestCase):
    """
    Test class for theano autocorrelation function
    """

    def setUp(self):
        np.random.seed(2015)


    def test_autocorrelation(self):
        """ Tests that the legacy and fast ac implementations produce identical output

        :returns: None
        :rtype: None
        """
        gaussian = Gaussian()
        sample_df = sample_to_df(MarkovJumpHMC, gaussian, num_steps=1000)
        slow_ac = autocorrelation(sample_df)
        fast_ac = normed_autocorrelation(sample_df)
        self.assertTrue((slow_ac == fast_ac).all())
