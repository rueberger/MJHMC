"""
This module contains the objective function used by Spearmint
  when searching for good hyperparameters.
In short, the objective function is the Re(r) where
  r is a complex number such that exp(r * t) is the
  best fit to the autocorrelation
"""

import numpy as np
from scipy.optimize import curve_fit
from mjhmc.misc.autocor import calculate_autocorrelation
from mjhmc.misc.plotting import plot_fit, plot_search_ac

grad_evals = {
    'Gaussian' : int(2E5),
    'RoughWell' : int(5E4),
    'MultimodalGaussian' : int(2E5)
}

debug = True


def obj_func(sampler, distr, job_id, **kwargs):
    """ Scores the performance of sampler on distribution given parameters

    :param sampler: sampler being tested. instance of mjhmc.samplers.markov_jump_hmc.HMCBase
    :param distr: distribution being used. instance of mjhmc.misc.distributions.Distribution
    :param job_id: integer label for job being run
    :returns: the score
    :rtype: float
    """
    cos_coef, normed_n_grad_evals, exp_coef, autocor, kwargs = obj_func_helper(sampler, distr, True, kwargs)
    if debug:
        plot_fit(normed_n_grad_evals, autocor, exp_coef, cos_coef, job_id, kwargs)
    return exp_coef

def obj_func_helper(sampler, distr, unpack, kwargs):
    """ Helper function for the objective function

    :param sampler: sampler being tested. instance of mjhmc.samplers.markov_jump_hmc.HMCBase
    :param distr: distribution being used. instance of mjhmc.misc.distributions.Distribution
    :param unpack: boolean flag of whether to unpack params or not.
    :param kwargs: dictionary of kwargs passed from parent function
    :returns: parameters of fitted curves, graph data for computed autocorrelation, kwargs
    :rtype: tuple

    """
    num_target_grad_evals =  grad_evals[type(distr).__name__]
    default_args = {
        "num_grad_steps": num_target_grad_evals,
        "sample_steps": 1,
        "num_steps": None,
        "half_window": True
    }
    if unpack:
        kwargs = unpack_params(kwargs)
    kwargs.update(default_args)

    print "Calculating autocorrelation for {} grad evals".format(num_target_grad_evals)
    ac_df = calculate_autocorrelation(sampler, distr, **kwargs)
    n_grad_evals = ac_df['num grad'].values.astype(int)
    # necessary to keep curve_fit from borking
    normed_n_grad_evals = n_grad_evals / (0.5 * num_target_grad_evals)
    autocor = ac_df['autocorrelation'].values
    print "Fitting curve"
    exp_coef, cos_coef = fit(normed_n_grad_evals.copy(), autocor.copy())
    return cos_coef, normed_n_grad_evals, exp_coef, autocor, kwargs


def min_idx(ac_df, target):
    ac_df.index = ac_df['num grad']
    ac_trunc = ac_df.loc[:, 'autocorrelation'] < target
    small_ac = ac_trunc[ac_trunc]
    if len(small_ac) != 0:
        return small_ac.index[0]
    else:
        return None

def unpack_params(params):
    """
    Spearmint passes params as 1x1 arrays.
    returns unpacked dict params
    probably not a problem, but just in case
    """
    unpacked_params = {}
    for key, item in params.iteritems():
        unpacked_params[key] = item[0]
    return unpacked_params

def fit(t, y):
    # very fast way to check for nan
    if not np.isnan(np.sum(y)):
        # used to truncate last 50 values for noise purposes
        # might still be worth doing?
        opt_params = curve_fit(curve, t, y)[0]
        return opt_params
    else:
        return 1E2


def curve(n, a, b):
    return np.exp(a * n) * np.cos(b * n)
