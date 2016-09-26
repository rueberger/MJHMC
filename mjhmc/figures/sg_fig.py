"""
This module contains a script for generating the spectral gap figure
Figure 3 in the paper (3rd revision on arXiv)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from scipy.linalg import eig

import time

import pandas as pd

from os.path import expanduser

from mjhmc.samplers.algebraic_hmc import (AlgebraicHMC, AlgebraicDiscrete,
                                    AlgebraicContinuous, AlgebraicReducedFlip)
from mjhmc.experiments.spectral import fit_inv_pdf, ladder_numerical_err_hist

# green blue palette
sns.set_palette("cubehelix", n_colors=2)
sns.set_context("talk")
sns.set_style("whitegrid", {"axes.linewidth": .5})


def sg(algebraic_sampler, full):
    """
    returns the spectral gap of the sampler object
    """
    T = algebraic_sampler.calculate_true_transition_matrix(full)
    w, v = eig(T)
    w_ord = np.sort(w)[::-1]
    if np.around(np.real_if_close(w_ord[0]), decimals=5) != 1:
        raise Exception("no eval with value 1")
    return 1 - np.absolute(w_ord[1])


def plot_spectral_gaps(max_n_dims, n_trials=25,
                       full=False, save_directory='~/tmp/figs/mjhmc'):
    """ Generates the spectral gap figure

    :param max_n_dims: max number of dimensions to go up to
    :param n_trials: number of trials averaged at each dimension
    :param full: True for computing the spectral gap of the full transition matrix (an 2*n_dim X 2*n_dim matrix)
    :param save_directory: path to save figure to
    :returns: None, saves a figure at the specified path
    :rtype: None
    """
    print("Computing empirical energy distribution")
    energy_hist, _  = ladder_numerical_err_hist()
    inv_pdf = fit_inv_pdf(energy_hist)
    hmc_sg = []
    rf_sg = []
    sgs = []
    t_begin = time.clock()
    orders = np.arange(3, max_n_dims) * 2
    for order in orders:
        t_start = time.clock()
        hmc_trials = []
        rf_trials = []
        for _ in xrange(n_trials):
            H = inv_pdf(np.random.random(order / 2))
            hmc = AlgebraicHMC(order, energies=H)
            rf = AlgebraicReducedFlip(order, energies=H)
            hmc_trials.append(sg(hmc, full))
            rf_trials.append(sg(rf, full))
        hmc_sg.append(hmc_trials)
        rf_sg.append(rf_trials)
        print "order {} took {} seconds".format(order, time.clock() - t_start)
    hmc_sg = np.array(hmc_sg)
    rf_sg = np.array(rf_sg)
    # putting into dataframe for seaborn
    for idx in xrange(n_trials):
        hmc_df = pd.DataFrame(dict(
            Sampler=["Discrete-time HMC"] * len(orders),
            subj=["subj{}".format(idx)] * len(orders),
            order=orders,
            sg=hmc_sg[:,idx]), dtype=np.float)
        rf_df = pd.DataFrame(dict(
            Sampler=["Markov Jump HMC"] * len(orders),
            subj=["subj{}".format(idx)] * len(orders),
            order=orders,
            sg=rf_sg[:,idx]), dtype=np.float)
        sgs.append(hmc_df)
        sgs.append(rf_df)
    sgs_df = pd.concat(sgs)


    print "computation finished. total time elapsed: {}".format(time.clock() - t_begin)
    sns.tsplot(sgs_df, time="order", unit="subj", condition="Sampler", value="sg")
    plt.ylabel("Spectral gap (log)")
    plt.xlabel("Number of states in ladder")
    plt.yscale('log')
    plt.title("Spectral gap vs. number of system states")
    if full:
        plt.savefig("{}/sg_gap_full_{}_energies_{}_trials.pdf".format(
            expanduser(save_directory), max_n_dims, n_trials))
    else:
        plt.savefig("{}/sg_gap_half_{}_energies_{}_trials.pdf".format(
            expanduser(save_directory), max_n_dims, n_trials))
