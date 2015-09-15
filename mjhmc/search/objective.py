import numpy as np
from scipy.optimize import curve_fit
from mjhmc.misc.autocor import calculate_autocorrelation
from mjhmc.misc.plotting import plot_fit, plot_search_ac

grad_evals = {
    'Gaussian' : int(2E5),
    'RoughWell' : int(2E6),
    'MultimodalGaussian' : int(2E5)
}

debug = True
use_exp = False


def obj_func(sampler, distr, job_id, **kwargs):
    num_target_grad_evals =  grad_evals[type(distr).__name__]
    default_args = {
        "num_grad_steps": num_target_grad_evals,
        "sample_steps": 1,
        "num_steps": None,
        "half_window": True
    }
    kwargs = unpack_params(kwargs)
    kwargs.update(default_args)
    ac_df = calculate_autocorrelation(sampler, distr, **kwargs)
    n = ac_df['num grad'].values.astype(int)
    # necessary to keep curve_fit from borking
    normed_n = n / (0.5 * num_target_grad_evals)
    y = ac_df['autocorrelation'].values
    if use_exp:
        # wtf fit is mutating input somehow
        r1 = fit(normed_n.copy(), y.copy())[0]
        if debug:
            plot_fit(normed_n, y, r1, job_id, kwargs)
        return -r1
    else:
        if np.isnan(np.sum(ac_df.autocorrelation.values)):
            return 11 * num_target_grad_evals
        for trial, target in enumerate(np.arange(0, 1, 0.1)):
            score = min_idx(ac_df, target)
            if score is not None:
                score += trial * num_target_grad_evals * 0.5
                break
        if debug:
            plot_search_ac(ac_df['num grad'].values.astype(int),
                           ac_df.autocorrelation.values,
                           job_id,
                           kwargs, score)
        return score or 5 * num_target_grad_evals

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
        opt_params, opt_cov = curve_fit(curve, t[:-50], y[:-50])
        return opt_params
    else:
        return 1E2

# def curve(n, r1, r2, s1, s2):
def curve(n, r):
    # return abs(s1) * np.exp(-r1 * n) + abs(s2) * np.exp(-r2 * n)
    return np.exp(-r, n)
