"""
Module containing spectral gap experiments
"""
import numpy as np
from scipy.interpolate import UnivariateSpline

from mjhmc.samplers.markov_jump_hmc import ControlHMC
from mjhmc.misc.distributions import Gaussian


def ladder_numerical_err_hist(distr=None, n_steps=int(1e5)):
    """ Compute a histogram of the numerical integration error
    on the state ladder. Implicitly assumes that such a distribution exists
    and is shared by all ladders

    Args:
     distr: distribution object to run on, make sure n_batch is big

    Returns:
     energies = {E(L^j \zeta) : j \in {0, ..., k}}^{n_batch}
    """
    distr = distr or Gaussian(nbatch=1)
    sampler = ControlHMC(distribution=distr)

    # [[ladder_energies]]
    energies = []
    r_counts = [0]
    ladder_energies = [np.squeeze(sampler.state.H())]
    for _ in range(n_steps):
        if sampler.r_count == r_counts[-1]:
            ladder_energies.append(np.squeeze(sampler.state.H()))
        else:
            energies.append(np.array(ladder_energies))
            ladder_energies = [np.squeeze(sampler.state.H())]
        r_counts.append(sampler.r_count)
        sampler.sampling_iteration()
    centered_energies = []
    for ladder_energies in energies:
        centered_energies += list(ladder_energies - ladder_energies[0])
    return centered_energies

def fit_inv_pdf(ladder_energies):
    """ Fit an interpolant to the inverse pdf of ladder energies
    Nasty hack to allow drawing energies from arbitrary distributions of ladder_energies

    Args:
      ladder_energies: array, output of ladder_numerical_err_hist

    Returns:
      interp_inv_pdf: inverse pdf interpolant
    """
    hist, bin_edges = np.histogram(ladder_energies, bins='auto')
    bin_mdpts = (np.diff(bin_edges) / 2) + bin_edges[:-1]
    # interpolation backward using slope bin_mdpts[1] - bin_mdpts[0]
    zero_interp_mdpt = -2 * bin_mdpts[0] + bin_mdpts[1]
    pdf = np.cumsum(hist) / np.sum(hist)
    # we add zero so that the interpolation is defined everywhere on [0, 1]
    pdf = np.concatenate([[0], pdf])
    bin_mdpts = np.concatenate([[zero_interp_mdpt], bin_mdpts])
    return UnivariateSpline(pdf, bin_mdpts, bbox=[0, 1], k=1)
