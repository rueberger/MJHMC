"""
This file contains various plotting utilities
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from mjhmc.samplers.markov_jump_hmc import Control, ContinuousTimeHMC, MarkovJumpHMC
from mjhmc.samplers.generic_discrete import Gibbs, Continuous, IsingState
from .autocor import calculate_autocorrelation

import numpy as np

# some formatting for seaborn
sns.set_palette("deep", desat=.6)
sns.set_context(rc={"figure.figsize": (8, 4)})

# interactive mode for ipython usage

def plot_search_ac(t, ac, job_id, params, score):
    plt.plot(t, ac)
    plt.title("Score: {}, beta: {}, epsilon: {}, M: {}".format(score,
        params['beta'], params['epsilon'], params['num_leapfrog_steps']))
    plt.savefig("job_{}_ac.png".format(job_id))

def plot_fit(t, y, r1, job_id, params):
    plt.plot(t, y, label='observed')
    # fitted = abs(s1) * np.exp(-r1 * t) + abs(s2) * np.exp(-r2 * t)
    fitted = np.exp(-r1 * t)
    plt.plot(t, fitted, label="fittted")
    plt.title("R: {}, beta: {}, epsilon: {}, M: {}".format(
        r1, params['beta'], params['epsilon'], params['num_leapfrog_steps']))
    plt.legend()
    plt.savefig("job_{}_fit.png".format(job_id))


def hist_1d(distr, nsamples=1000, nbins=250):
    """
    plots a 1d histogram from each sampler
    distr is (an unitialized) class from distributions
    """
    distribution = distr(ndims=1)
    control = Control(distribution.Xinit, distribution.E, distribution.dEdX)
    experimental = ContinuousTimeHMC(distribution.Xinit, distribution.E, distribution.dEdX)

    plt.hist(control.sample(nsamples)[0], nbins, normed=True, label="Standard Control", alpha=.5)
    plt.hist(experimental.sample(nsamples)[0], nbins, normed=True, label="Continuous-time Control",alpha=.5)
    plt.legend()

def gauss_1d(nsamples=1000, nbins=250):
    """
    Simple test plots.
    Draws nsamples with sampler from a (well conditioned unit variance)
    gaussian and plots a histogram of both of them
    """
    hist_1d(misc.distributions.TestGaussian)


def gauss_2d(nsamples=1000):
    """
    Another simple test plot
    1d gaussian sampled from each sampler visualized as a joint 2d gaussian
    """
    gaussian = misc.distributions.TestGaussian(ndims=1)
    control = Control(gaussian.Xinit, gaussian.E, gaussian.dEdX)
    experimental = ContinuousTimeHMC(gaussian.Xinit, gaussian.E, gaussian.dEdX)

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
    d_ac = calculate_autocorrelation(Control, distr, num_steps, sample_steps)
    c_ac = calculate_autocorrelation(ContinuousTimeHMC, distr, num_steps, sample_steps)
    d_ac.index = d_ac['num grad']
    c_ac.index = c_ac['num grad']
    # combine plots, maybe join possible after all
    d_ac['autocorrelation'].plot(label='discrete')
    c_ac['autocorrelation'].plot(label='continuous')
    plt.legend()

def plot_ac_ising(J=None, ndims=10, nbatch=100, num_steps=1000):
    """
    Autocorrelation for the ising model
    if J is none, generates one randomly
    """
    ising = IsingState(ndims, nbatch, J)
    g = Gibbs(ising)
    c = Continuous(ising)
    g_ac = autocorrelation(g.sample(num_steps, True))
    c_ac = autocorrelation(c.sample(num_steps, True))
    plt.plot(g_ac['num energy'], g_ac['autocorrelation'], label="Gibbs")
    plt.plot(c_ac['num energy'], c_ac['autocorrelation'], label="Continuous")
    plt.legend()
