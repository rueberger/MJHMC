"""
Script to plot the autocorrelation and fit for previously found optimal parameters
"""

from mjhmc.search.objective import obj_func_helper
from mjhmc.figures.ac_fig import load_params
from mjhmc.samplers.markov_jump_hmc import ControlHMC, MarkovJumpHMC
from mjhmc.misc.distributions import RoughWell, Gaussian, MultimodalGaussian
from mjhmc.misc.plotting import plot_fit

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def plot_all_best(custom_params=None):
    """ Creates a plot with the autocorrelation and fit for each distribution and sampler

    :param custom_params: dictionary of custom params will be used on all distributions
      and samplers. if None uses the current best params for each
    :returns: None
    :rtype: None

    """
    distributions = [
        RoughWell(nbatch=200)
        # Gaussian(ndims=10, nbatch=200),
        # MultimodalGaussian(ndims=5, separation=1)
    ]
    samplers = [
        ControlHMC,
        MarkovJumpHMC
    ]
    with PdfPages("validation.pdf") as pdf:
        for distribution in distributions:
            # [control, mjhmc, lahmc]
            if custom_params is None:
                params = load_params(distribution)
            else:
                params = [custom_params] * 3
            active_params = params[:-1]
            for sampler, hparams in zip(samplers, active_params):
                print("Now running for {} on {}".format(sampler, distribution))
                cos_coef, n_grad_evals, exp_coef, autocor, _ = obj_func_helper(
                    sampler, distribution.reset(), False, hparams)
                fig = plot_fit(n_grad_evals,
                               autocor,
                               exp_coef,
                               cos_coef,
                               'validation',
                               hparams,
                               save=False
                           )
                pdf.savefig(fig)

def plot_comparison(samplers, params, distribution):
    """ Plot a comparison between samplers and params

    :param samplers: list of samplers to test
    :param params: respective list of parameters for each sampler
    :param distribution: distribution to compare on
    :returns: None
    :rtype: None
    """
    for sampler, hparams in zip(samplers, params):
        _, n_grad_evals, _, autocor, _ = obj_func_helper(
            sampler, distribution.reset(), False, hparams)
        plt.plot(n_grad_evals, autocor,
                 label="B: {}, eps: {}, M: {}".format(hparams['beta'],
                                                      hparams['epsilon'],
                                                      hparams['num_leapfrog_steps']))
    plt.legend()
    plt.savefig('comparison.pdf')
