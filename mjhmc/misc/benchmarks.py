""" This module contains some utilities for benchmarking and profiling
"""

import matplotlib.pyplot as plt
import seaborn as sns
plt.ion()

import time

from mjhmc.search.objective import obj_func
from mjhmc.samplers.markov_jump_hmc import MarkovJumpHMC
from mjhmc.misc.distributions import ProductOfT, Gaussian
from scipy.sparse import rand
import numpy as np

#specify parameters
# normalize cost to sample

def benchmark_batch_size_scaling(sampler_cls=MarkovJumpHMC,
                                 distribution_cls=Gaussian, ndims=2,
                                 n_samples=1000, n_itrs=10):
    """ Plots run time against batch size

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

        times = np.zeros(n_itrs)
        for idx in range(n_itrs):
            t_i = time.time()
            # argument error here
            _ = sampler.sample(nsamples=n_samples)
            t_f = time.time()
            times[idx] = t_f - t_i
        run_time.append(np.mean(times))
    plt.plot(batch_sizes, run_time)
    plt.show()
