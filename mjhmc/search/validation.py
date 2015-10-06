"""
Script to plot the autocorrelation and fit for previously found optimal parameters
"""

from mjhmc.search.objective import obj_func_helper
from mjhmc.figures.ac_fig import load_params
from mjhmc.samplers.markov_jump_hmc import ControlHMC, MarkovJumpHMC
from mjhmc.misc.distributions import RoughWell, Gaussian, MultimodalGaussian
from mjhmc.misc.plotting import plot_fit

from matplotlib.backends.backend_pdf import PdfPages

def plot_all_best():
    """ Creates a plot with the autocorrelation and fit for each distribution and sampler
    Saves all plots to a pdf

    :returns: None
    :rtype: None
    """
    distributions = [
        RoughWell(nbatch=200),
        Gaussian(ndims=10, nbatch=200),
        MultimodalGaussian(ndims=5, separation=1)
    ]
    samplers = [
        ControlHMC,
        MarkovJumpHMC
    ]
    with PdfPages("validation.pdf") as pdf:
        for distribution in distributions:
            # [control, mjhmc, lahmc]
            params = load_params(distribution)
            active_params = params[:-1]
            for sampler, hparams in zip(samplers, active_params):
                print "Now running for {} on {}".format(sampler, distribution)
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
