"""
This file contains various plotting utilities
"""

#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


from mjhmc.samplers.markov_jump_hmc import HMCBase, ContinuousTimeHMC, MarkovJumpHMC
from mjhmc.misc.distributions import TestGaussian
from .autocor import calculate_autocorrelation

import numpy as np

# some formatting for seaborn
sns.set_palette("deep", desat=.6)
sns.set_context(rc={"figure.figsize": (8, 4)})

#pylint: disable=too-many-arguments

def plot_search_ac(t, ac, job_id, params, score):
    plt.plot(t, ac)
    plt.title("Score: {}, beta: {}, epsilon: {}, M: {}".format(score,
        params['beta'], params['epsilon'], params['num_leapfrog_steps']))
    plt.savefig("job_{}_ac.png".format(job_id))

def plot_fit(grad_evals, autocor, exp_coef, cos_coef, job_id, params, save=True):
    """ Debug plot for hyperparameter search
    Saves a plot of the autocorrelation and the fitted complex exponential of the form
    ac(t) = exp(a * t) * cos(b * t)

    :param grad_evals: array of grad evals. Use as x-axis
    :param autocor: array of autocor. Of same shape as grad_evals
    :param exp_coef: float. Parameter a for fitted func
    :param cos_coef: float. Parameter b for fitted func
    :param job_id: the id of the Spearmint job that tested these params
    :param params: the set of parameters for this job
    :returns: the rendered figure
    :rtype: plt.figure()
    """
    fig = plt.figure()
    plt.plot(grad_evals, autocor, label='observed')
    fitted = np.exp(exp_coef * grad_evals) * np.cos(cos_coef * grad_evals)
    plt.plot(grad_evals, fitted, label="fittted")
    plt.title("Score: {}, beta: {}, epsilon: {}, M: {}".format(
        exp_coef, params['beta'], params['epsilon'], params['num_leapfrog_steps']))
    plt.legend()
    if save:
        plt.savefig("job_{}_fit.pdf".format(job_id))
    return fig


def hist_1d(distr, nsamples=1000, nbins=250):
    """
    plots a 1d histogram from each sampler
    distr is (an unitialized) class from distributions
    """
    distribution = distr(ndims=1)
    #control = HMCBase(distribution=distribution)
    experimental = MarkovJumpHMC(distribution=distribution, resample=False, epsilon=1)

    #plt.hist(control.sample(nsamples)[0], nbins, normed=True, label="Standard HMCBase", alpha=.5)
    plt.hist(experimental.sample(nsamples)[0], nbins, normed=True, label="Continuous-time HMCBase",alpha=.5)
    plt.legend()

def gauss_1d(nsamples=1000, nbins=250):
    """
    Simple test plots.
    Draws nsamples with sampler from a (well conditioned unit variance)
    gaussian and plots a histogram of both of them
    """
    hist_1d(TestGaussian, nsamples, nbins)
    test_points = np.linspace(4, -4, 1000)
    true_curve = (1. / np.sqrt(2*np.pi)) * np.exp(- (test_points**2) / 2.)
    plt.plot(test_points, true_curve, label="True curve")
    plt.legend()


def gauss_2d(nsamples=1000):
    """
    Another simple test plot
    1d gaussian sampled from each sampler visualized as a joint 2d gaussian
    """
    gaussian = TestGaussian(ndims=1)
    control = HMCBase(distribution=gaussian)
    experimental = MarkovJumpHMC(distribution=gaussian, resample=False)


    with sns.axes_style("white"):
        sns.jointplot(
            control.sample(nsamples)[0],
            experimental.sample(nsamples)[0],
            kind='hex',
            stat_func=None)

def hist_2d(distribution, nsamples, **kwargs):
    """
    Plots a 2d hexbinned histogram of distribution
    """
    distr = distribution(ndims=2)
    sampler = MarkovJumpHMC(distr.Xinit, distr.E, distr.dEdX, **kwargs)
    samples = sampler.sample(nsamples)

    with sns.axes_style("white"):
        sns.jointplot(samples[0], samples[1], kind='kde', stat_func=None)


def jump_plot(distribution, nsamples=100, **kwargs):
    """
    Plots samples drawn from distribution with dwelling time on the x-axis
    and the sample value on the y-axis
    1D only
    """
    distr = distribution(ndims=1, nbatch=1, **kwargs)
    # sampler = MarkovJumpHMC(np.array([0]).reshape(1,1),
    sampler = MarkovJumpHMC(distr.Xinit,
                            distr.E, distr.dEdX,
                            epsilon=.3, beta=.2, num_leapfrog_steps=5)
    x_t = []
    d_t = []
    transitions = []
    last_L_count, last_F_count, last_R_count = 0, 0, 0
    for idx in xrange(nsamples):
        sampler.sampling_iteration()
        x_t.append(sampler.state.X[0, 0])
        d_t.append(sampler.dwelling_times[0])
        if sampler.L_count - last_L_count == 1:
            transitions.append("L")
            last_L_count += 1
        elif sampler.F_count - last_F_count == 1:
            transitions.append("F")
            last_F_count += 1
        elif sampler.R_count - last_R_count == 1:
            transitions.append("R")
            last_R_count += 1
    t = np.cumsum(d_t)
    plt.scatter(t, x_t)
    t = np.array(t).reshape(len(t), 1)
    x_t = np.array(x_t).reshape(len(x_t), 1)
    transitions = np.array(transitions).reshape(len(transitions), 1)
    data = np.concatenate((x_t, t, transitions), axis=1)
    return pd.DataFrame(data, columns=['x', 't', 'transitions'])



def plot_autocorrelation(distribution, ndims=2,
                         num_steps=200, nbatch=100, sample_steps=10):
    """
    plot autocorrelation versus gradient evaluations

    set to be deprecated
    """
    distr = distribution(ndims, nbatch)
    d_ac = calculate_autocorrelation(HMCBase, distr, num_steps, sample_steps)
    c_ac = calculate_autocorrelation(ContinuousTimeHMC, distr, num_steps, sample_steps)
    d_ac.index = d_ac['num grad']
    c_ac.index = c_ac['num grad']
    # combine plots, maybe join possible after all
    d_ac['autocorrelation'].plot(label='discrete')
    c_ac['autocorrelation'].plot(label='continuous')
    plt.legend()
