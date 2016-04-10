"""
 This module contains methods for generating and caching fair initializations for MJHMC
"""
import pickle
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
    print('Generating fair initialization for {} by burning in {} steps'.format(
        distribution.__name__, BURN_IN_STEPS)
    assert distribution.nbatch == MAX_N_PARTICLES
    mjhmc = MarkovJumpHMC(distribution=distribution)
    for _ in xrange(BURN_IN_STEPS):
        mjhmc.sampling_iteration()
    fair_x = mjhmc.state.copy().X
    fair_v = mjhmc.state.copy().V
    # check that i'm the right shape
    return np.hstack((fair_x, fair_v))

def cache_initialization(distribution):
    """ Generates fair initialization for mjhmc on distribution and then caches it

    :param distribution: Distribution object. Must have nbatch == MAX_N_PARTICLES
    :returns:
    :rtype:
    """
    distr_name = distribution.__name__
    distr_hash = hash(distribution)
    fair_init = generate_initialization(distribution)
    file_name = '../../initializations/{}_{}.pickle'.format(distr_name, distr_hash)
    with open(file_name, 'wb') as f:
        pickle.dump(fair_init, f)
    print "Fair initialization for {} saved as {}".format(distr_name, file_name)
