"""
Module containing various experiments
"""
import numpy as np

from mjhmc.samplers.markov_jump_hmc import ControlHMC
from mjhmc.misc.distributions import Gaussian


def numerical_error_distr(distr=None, n_batch=5000, n_steps=int(1e4)):
    """ Compute a histogram of the numerical integration error
     of a particle on the ladder.

    Args:
     distr: distribution object to run on, make sure n_batch is big

    Returns:
     energies = {E(L^j \zeta) : j \in {0, ..., k}}^{n_batch}
    """
    distr = distr or Gaussian(nbatch=n_batch)
    sampler = ControlHMC(distribution=distr)
    state = sampler.state
    energies = np.zeros((n_steps, n_batch))
    for step_idx in range(n_steps):
        energies[step_idx, :] = np.squeeze(state.H())
    energies -= energies[0, :]
    return energies
