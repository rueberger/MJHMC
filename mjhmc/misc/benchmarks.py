""" This module contains some utilities for benchmarking and profiling
"""

import matplotlib.pyplot as plt
import seaborn as sns
plt.ion()

import time


from mjhmc.samplers.markov_jump_hmc import MarkovJumpHMC
from mjhmc.misc.distributions import ProductOfT, Gaussian
from mjhmc.misc.autocor import calculate_autocorrelation

import numpy as np

#specify parameters
# normalize cost to sample

def benchmark_batch_size_scaling(sampler_cls=MarkovJumpHMC,
                                 distribution_cls=Gaussian, ndims=2,
                                 n_samples=1000, n_itrs=10):
    """ Plots run time against batch size
    normalizes to time per sample

    :param sampler_cls: sampler to test
    :param distribution_cls: distribution to test on
    :param ndims: dimension of the space, passed to distribution_cls upon instantiation
    :param n_samples: number of samples to generate
    :param n_itrs: number of trials to average
    :returns: None, makes a plot
    :rtype: None

    """
    run_time = []
    batch_sizes = np.arange(1, 500, 25)
    for n_batch in batch_sizes:
        distribution = distribution_cls(ndims=ndims, nbatch=n_batch)
        sampler = sampler_cls(distribution=distribution)
        print "now doing batch size {}".format(n_batch)
        times = np.zeros(n_itrs)
        for idx in range(n_itrs):
            t_i = time.time()
            _ = sampler.sample(n_samples=n_samples)
            t_f = time.time()
            times[idx] = t_f - t_i
        run_time.append(np.mean(times) / (n_samples * n_batch))
    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.plot(batch_sizes, run_time)
    axis.set_yscale('log')
    axis.set_ylabel("log seconds per sample")
    axis.set_xlabel("batch size")
    axis.set_title("{}: compute time per sample versus batch size".format(sampler_cls.__name__))

def benchmark_ac_batch_size_scaling(sampler_cls=MarkovJumpHMC,
                                    distribution_cls=Gaussian, ndims=2,
                                    n_samples=1000, n_itrs=10):
    """ Plots run time of sampling and calculating the autocorrelation of n_samples (fixed)
      against batch size
    normalizes to time per sample

    :param sampler_cls: sampler to test
    :param distribution_cls: distribution to test on
    :param ndims: dimension of the space, passed to distribution_cls upon instantiation
    :param n_samples: number of samples to generate
    :param n_itrs: number of trials to average
    :returns: None, makes a plot
    :rtype: None

    """
    run_time = []
    batch_sizes = np.arange(1, 500, 25)
    for n_batch in batch_sizes:
        distribution = distribution_cls(ndims=ndims, nbatch=n_batch)
        print "now doing batch size {}".format(n_batch)
        times = np.zeros(n_itrs)
        for idx in range(n_itrs):
            t_i = time.time()
            _ = calculate_autocorrelation(sampler_cls, distribution, num_steps=n_samples, half_window=True)
            t_f = time.time()
            times[idx] = t_f - t_i
        run_time.append(np.mean(times) / (n_samples * n_batch))
    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.plot(batch_sizes, run_time)
    axis.set_yscale('log')
    axis.set_ylabel("log seconds per sample")
    axis.set_xlabel("batch size")
    axis.set_title("{} autocorrelation: compute time per sample versus batch size".format(sampler_cls.__name__))

def autocorrelation_vs_n_samples(sampler_cls=MarkovJumpHMC,
                                 distribution_cls=Gaussian, ndims=2,
                                n_batch=200, n_itrs=10):
    """ Plots run time of sampling and calculating the autocorrelation against n_samples,
    fixing batch size
    plots total time - not time per sample

    :param sampler_cls: sampler to test
    :param distribution_cls: distribution to test on
    :param ndims: dimension of the space, passed to distribution_cls upon instantiation
    :param n_samples: number of samples to generate
    :param n_itrs: number of trials to average
    :returns: None, makes a plot
    :rtype: None

    """
    run_time = []
    sample_sizes = np.arange(100, 10000, 250)
    for n_samples in sample_sizes:
        distribution = distribution_cls(ndims=ndims, nbatch=n_batch)
        print "now running for {} samples".format(n_samples)
        times = np.zeros(n_itrs)
        for idx in range(n_itrs):
            t_i = time.time()
            _ = calculate_autocorrelation(sampler_cls, distribution, num_steps=n_samples, half_window=True)
            t_f = time.time()
            times[idx] = t_f - t_i
        print "last run took time {}".format(times[-1])
        run_time.append(np.mean(times))
    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.plot(sample_sizes, run_time)
    axis.set_yscale('log')
    axis.set_ylabel("log seconds per sample")
    axis.set_xlabel("batch size")
    axis.set_title("{}: autocorrelation compute time by number of samples".format(sampler_cls.__name__))