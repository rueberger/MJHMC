"""
 This module contains methods for generating and caching fair initializations for MJHMC
"""
import pickle
import numpy as np
from copy import deepcopy
from mjhmc.samplers.markov_jump_hmc import MarkovJumpHMC

BURN_IN_STEPS = 100
#BURN_IN_STEPS = int(1E6)
VAR_STEPS = int(1E4)
MAX_N_PARTICLES = 1000

def generate_initialization(distribution):
    """ Run mjhmc for BURN_IN_STEPS on distribution, generating a fair set of initial states

    :param distribution: Distribution object. Must have nbatch == MAX_N_PARTICLES
    :returns: a set of fair initial states and an estimate of the variance
    :rtype: tuple: (array of shape (distribution.ndims, MAX_N_PARTICLES), float)
    """
    print('Generating fair initialization for {} by burning in {} steps'.format(
        type(distribution).__name__, BURN_IN_STEPS))
    assert distribution.nbatch == MAX_N_PARTICLES
    mjhmc = MarkovJumpHMC(distribution=distribution)
    for _ in xrange(BURN_IN_STEPS):
        mjhmc.sampling_iteration()
    mjhmc.resample = False
    samples = mjhmc.sample(n_samples=VAR_STEPS)
    var_estimate = np.var(samples)
    # we discard v since p(x,v) = p(x)p(v)
    fair_x = mjhmc.state.copy().X
    return (fair_x, var_estimate)

def cache_initialization(distribution):
    """ Generates fair initialization for mjhmc on distribution and then caches it

    :param distribution: Distribution object. Must have nbatch == MAX_N_PARTICLES
    :returns:
    :rtype:
    """
    distr_name = type(distribution).__name__
    distr_hash = hash(distribution)
    fair_init, var_estimate = generate_initialization(distribution)
    file_name = '../../initializations/{}_{}.pickle'.format(distr_name, distr_hash)
    with open(file_name, 'wb') as cache_file:
        pickle.dump((fair_init, var_estimate), cache_file)
    print "Fair initialization for {} saved as {}".format(distr_name, file_name)
    print "The embedded jump process on {} has estimated variance of {}".format(
        distr_name, var_estimate)

    # hack to estimate variance of the distribution itself
    distr_copy = deepcopy(distribution)
    distr_copy.nbatch = VAR_STEPS
    distr_copy.gen_init_X()
    true_var_estimate = np.var(distr_copy.Xinit)
    print "Meanwhile {} itself has an estimated variance of {}".format(
        distr_name, true_var_estimate)
