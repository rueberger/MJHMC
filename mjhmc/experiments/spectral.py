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
     run_lengths: list of observed ladder sizes
    """
    distr = distr or Gaussian(nbatch=1)
    sampler = ControlHMC(distribution=distr)

    # [[ladder_energies]]
    energies = []
    run_lengths = []
    r_counts = [0]
    ladder_energies = [np.squeeze(sampler.state.H())]
    run_length = 0
    for _ in range(n_steps):
        if sampler.r_count == r_counts[-1]:
            run_length += 1
            ladder_energies.append(np.squeeze(sampler.state.H()))
        else:
            run_lengths.append(run_length)
            run_length = 0
            energies.append(np.array(ladder_energies))
            ladder_energies = [np.squeeze(sampler.state.H())]
        r_counts.append(sampler.r_count)
        sampler.sampling_iteration()
    centered_energies = []
    for ladder_energies in energies:
        centered_energies += list(ladder_energies - ladder_energies[0])
    return centered_energies, run_lengths

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

def ladder_generator(distribution, epsilon, num_leapfrog_steps, beta, max_steps=int(1e5)):
    """ Returns a generator over ladders encountered while sampling from
    the SparseImageCode distribution

    Args:
      distribution: the distribution to test. must have nbatch==1 - Distribution
      epsilon: integrator step size - float
      num_leapfrog_steps: number of integrator steps per L application - int
      beta: rate of momentum corruption - float
      max_steps: number of sampling steps to run the generator for - int

    Returns:
      ladder_generator: next returns an array of ladder energies of length (order / 2)
    """
    from mjhmc.samplers.markov_jump_hmc import MarkovJumpHMC
    from mjhmc.samplers.algebraic_hmc import StateGroup
    from mjhmc.misc.distributions import Distribution

    # max order for an individual ladder
    # error is thrown if ever exceeded
    MAX_ORDER = 100000

    # check that distribution is as required
    assert isinstance(distribution, Distribution)
    assert distribution.nbatch == 1

    mjhmc = MarkovJumpHMC(distribution=distribution,
                          epsilon=epsilon,
                          num_leapfrog_steps=num_leapfrog_steps,
                          beta=beta)
    # ignore first element
    steps_per_ladder = [0]
    last_r_count = 0
    last_l_count = 0
    last_f_count = 0
    ladder_group = StateGroup(MAX_ORDER, np.zeros(MAX_ORDER / 2))
    # initialized randomly otherwise
    ladder_group.state = [0, 0]
    ladder_group.energies[0] = np.squeeze(mjhmc.state.H())
    for _ in range(max_steps):
        mjhmc.sampling_iteration()
        # last operator was R
        if mjhmc.r_count != last_r_count:
            # increment r count
            last_r_count += 1
            # extract the observed ladder energies
            # ladder_group.energies is of the form
            # [e_0, e_1.... e_n, 0, 0, ..., e_{n+1},... e_m]
            forward_energies = []
            backwards_energies = []
            for e_x in ladder_group.energies:
                if e_x != 0:
                    forward_energies.append(e_x)
                else:
                    break
            for e_x in ladder_group.energies[::-1]:
                if e_x != 0:
                    backwards_energies.append(e_x)
                else:
                    break
            assert len(forward_energies) + len(backwards_energies) < (MAX_ORDER / 2)
            yield np.array(backwards_energies[::-1] + forward_energies)

            # reset ladder group
            ladder_group = StateGroup(MAX_ORDER, np.zeros(MAX_ORDER / 2))
            # initialized randomly otherwise
            ladder_group.state = [0, 0]
            ladder_group.energies[0] = np.squeeze(mjhmc.state.H())
            steps_per_ladder.append(last_r_count + last_l_count + last_f_count - steps_per_ladder[-1])
        # last operator was L
        elif mjhmc.l_count != last_l_count:
            last_l_count += 1
            ladder_group.L()
            ladder_group.energies[ladder_group.idx()] = np.squeeze(mjhmc.state.H())
        # last operator was F
        elif mjhmc.f_count != last_f_count:
            last_f_count += 1
            ladder_group.F()

def sp_img_ladder_generator(*args, **kwargs):
    from mjhmc.misc.tf_distributions import SparseImageCode
    sp_img = SparseImageCode(n_patches=1, n_batches=1)
    return ladder_generator(sp_img, *args, **kwargs)

def test_fig(max_steps=int(1e4), full=True):
    """ Simple test figure to make sure everything is good to go
    Poor mans integration test
    """
    from mjhmc.samplers.algebraic_hmc import AlgebraicHMC, AlgebraicReducedFlip
    from mjhmc.figures.sg_fig import sg
    import matplotlib.pyplot as plt

    ladders = [l for l in sp_img_ladder_generator(1e-3, 1, 0.01, max_steps=max_steps) if len(l) > 1]
    ladder_lens = [len(l) for l in ladders]

    print("Computing spectral gaps for {} ladders".format(len(ladders)))

    mjhmc_sgs = []
    control_sgs = []

    for ladder in ladders:
        algebraic_mjhmc = AlgebraicReducedFlip(len(ladder) * 2, energies=ladder)
        algebraic_hmc = AlgebraicHMC(len(ladder) * 2, energies=ladder)

        mjhmc_sgs.append(sg(algebraic_mjhmc, full))
        control_sgs.append(sg(algebraic_hmc, full))

    plt.scatter(ladder_lens, mjhmc_sgs, marker='x', label='MJHMC')
    plt.scatter(ladder_lens, mjhmc_sgs, marker='o', label='control')
    plt.legend()
    ax = plt.gca()
    ax.set_yscale('log')
    return mjhmc_sgs, control_sgs, ladder_lens
