import numpy as np
from scipy.optimize import curve_fit
from mjhmc.misc.autocor import calculate_autocorrelation
from mjhmc.misc.plotting import plot_fit, plot_search_ac

grad_evals = {
    'Gaussian' : int(2E5),
    'RoughWell' : int(1E3),
    'MultimodalGaussian' : int(2E5)
}

debug = True


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
    n_grad_evals = ac_df['num grad'].values.astype(int)
    # necessary to keep curve_fit from borking
    normed_n_grad_evals = n_grad_evals / (0.5 * num_target_grad_evals)
    autocor = ac_df['autocorrelation'].values
    exp_coef, cos_coef = fit(normed_n_grad_evals.copy(), autocor.copy())
    if debug:
        plot_fit(normed_n_grad_evals, autocor, exp_coef, cos_coef, job_id, kwargs)
    return exp_coef




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
