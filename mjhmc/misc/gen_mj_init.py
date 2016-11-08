"""
 This module contains methods for generating and caching fair initializations for MJHMC
"""

import pickle
import numpy as np
from mjhmc.samplers.markov_jump_hmc import MarkovJumpHMC, ControlHMC
from .utils import package_path

BURN_IN_STEPS = 1000
VAR_STEPS = 500
# BURN_IN_STEPS = int(2E6)
# VAR_STEPS = int(1E6)
MAX_N_PARTICLES = 1000

def generate_initialization(distribution):
    """ Run mjhmc for BURN_IN_STEPS on distribution, generating a fair set of initial states

    :param distribution: Distribution object. Must have nbatch == MAX_N_PARTICLES
    :returns: a set of fair initial states and an estimate of the variance for emc and true both
    :rtype: tuple: (array of shape (distribution.ndims, MAX_N_PARTICLES), float, float)
    """
    print('Generating fair initialization for {} by burning in {} steps'.format(
        type(distribution).__name__, BURN_IN_STEPS))
    assert BURN_IN_STEPS > VAR_STEPS
    # must rebuild graph to nbatch=MAX_N_PARTICLES
    if distribution.backend == 'tensorflow':
        distribution.build_graph()
    mjhmc = MarkovJumpHMC(distribution=distribution, resample=False)
    for _ in xrange(BURN_IN_STEPS - VAR_STEPS):
        mjhmc.sampling_iteration()
    assert mjhmc.resample == False

    emc_var_estimate, mjhmc = online_variance(mjhmc, distribution)
    # we discard v since p(x,v) = p(x)p(v)
    mjhmc_endpt = mjhmc.state.copy().X

    # otherwise will go into recursive loop
    distribution.mjhmc = False
    try:
        distribution.gen_init_X()
    except NotImplementedError:
        print("No explicit init method found, using mjhmc endpoint")

    distribution.E_count = 0
    distribution.dEdX_count = 0

    control = ControlHMC(distribution=distribution)
    for _ in xrange(BURN_IN_STEPS - VAR_STEPS):
        control.sampling_iteration()
    true_var_estimate, control = online_variance(control, distribution)
    control_endpt = control.state.copy().X

    return mjhmc_endpt, emc_var_estimate, true_var_estimate, control_endpt

def cache_initialization(distribution):
    """ Generates fair initialization for mjhmc on distribution and then caches it

    :param distribution: Distribution object. Must have nbatch == MAX_N_PARTICLES
    :returns:
    :rtype:
    """
    distr_name = type(distribution).__name__
    distr_hash = hash(distribution)
    mjhmc_endpt, emc_var_estimate, true_var_estimate, control_endpt = generate_initialization(distribution)

    file_name = '{}_{}.pickle'.format(distr_name, distr_hash)
    file_prefix = '{}/initializations'.format(package_path())
    with open('{}/{}'.format(file_prefix, file_name), 'wb') as cache_file:
        pickle.dump((mjhmc_endpt, emc_var_estimate, true_var_estimate, control_endpt), cache_file)
    print "Fair initialization for {} saved as {}".format(distr_name, file_name)
    print "The embedded jump process on {} has estimated variance of {}".format(
        distr_name, emc_var_estimate)
    print "Meanwhile {} itself has an estimated variance of {}".format(
        distr_name, true_var_estimate)


def online_variance(sampler, distribution):
    """ computes the variance in an online fashion to allow arbitrarily large sample sizes


    :param sampler: initialized sampler
    :param distribution: initialized distribution
    :returns: variance estimate, sampler (for convenience)
    :rtype: float, HMCBase

    """
    #online variance computation, algorithm due to Knuth and Wellford
    curr_mean = 0
    curr_sumsq = 0
    trial_idx = 0
    for _ in xrange(VAR_STEPS):
        # very slow but safe
        for val in  sampler.sample(1).ravel():
            trial_idx += 1
            delta = val - curr_mean
            curr_mean += float(delta) / trial_idx
            curr_sumsq += delta * (val - curr_mean)
    var_estimate = curr_sumsq / float(VAR_STEPS * distribution.nbatch * distribution.ndims - 1)
    return var_estimate, sampler
