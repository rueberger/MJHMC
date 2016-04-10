"""
 This module contains methods for generating and caching fair initializations for MJHMC
"""

import numpy as np
from mjhmc.samplers.markov_jump_hmc import MarkovJumpHMC

#BURN_IN_STEPS = int(1E6)
BURN_IN_STEPS = 10
MAX_N_PARTICLES = 1000

def generate_initialization(distribution):
    """ Run mjhmc for BURN_IN_STEPS on distribution, generating a fair set of initial states

    :param distribution: Distribution object. Must have nbatch == MAX_N_PARTICLES
    :returns: the set of fair initial states
    :rtype: array of shape (2 * distribution.ndims, MAX_N_PARTICLES)
    """
    assert distribution.nbatch == MAX_N_PARTICLES
    mjhmc = MarkovJumpHMC(distribution=distribution)
    for _ in xrange(BURN_IN_STEPS):
        mjhmc.sampling_iteration()
    fair_x = mjhmc.state.copy().X
    fair_v = mjhmc.state.copy().V
    # check that i'm the right shape
    return np.hstack((fair_x, fair_v))

def cache_initialization(distribution):
    """ Generates fair initialization for mjhcm on distribution and then caches it

    :param distribution: Distribution object. Must have nbatch == MAX_N_PARTICLES
    :returns:
    :rtype:
    """
    # what naming scheme?
