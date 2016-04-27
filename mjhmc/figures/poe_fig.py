"""
This module contains a script for generating the product of experts image patch figure
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # no displayed figures -- need to call before loading pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import itertools
from scipy.sparse import rand

from mjhmc.misc.distributions import ProductOfT
from mjhmc.samplers.markov_jump_hmc import MarkovJumpHMC, ControlHMC
from mjhmc.samplers.hmc_state import HMCState

from nuts import nuts6

plt.ion()


# for deterministic params for poet
np.random.seed(1234)

mjhmc_params = {'epsilon' : 0.127, 'beta' : .01,'num_leapfrog_steps' : 1}
control_params = {'epsilon' : 0.065, 'beta' : 0.01, 'num_leapfrog_steps' : 1}
# mjhmc_params = control_params

def generate_figure_samples(samples_per_frame, n_frames, burnin = int(1e4)):
    """ Generates the figure

    :param samples_per_frame: number of sample steps between each frame
    :param n_frames: number of frames to draw
    :returns: None
    :rtype: None
    """
    n_samples = samples_per_frame * n_frames
    ndims = 36
    nbasis = 72

    rand_val = rand(ndims,nbasis/2,density=0.25)
    W = np.concatenate([rand_val.toarray(), -rand_val.toarray()],axis=1)
    logalpha = np.random.randn(nbasis, 1)
    poe = ProductOfT(nbatch=1, W=W, logalpha=logalpha)

    ## NUTS uses a different number of grad evals for each update step!!
    ## makes it very hard to compare against others w/ same number of update steps
    # # NUTS
    # print "NUTS"
    # nuts_init = poe.Xinit[:, 0]
    # nuts_samples = nuts6(poe.reset(), n_samples, nuts_burnin, nuts_init)[0]
    # nuts_frames = [nuts_samples[f_idx * samples_per_frame, :] for f_idx in xrange(0, n_frames)]
    # x_init = nuts_samples[0, :].reshape(ndims, 1)

    ## burnin
    print "MJHMC burnin"
    x_init = poe.Xinit #[:, [0]]
    mjhmc = MarkovJumpHMC(distribution=poe.reset(), **mjhmc_params)
    mjhmc.state = HMCState(x_init.copy(), mjhmc)
    mjhmc_samples = mjhmc.sample(burnin)
    print mjhmc_samples.shape
    x_init = mjhmc_samples[:, [0]]

    # control HMC
    print "Control"
    hmc = ControlHMC(distribution=poe.reset(), **control_params)
    hmc.state = HMCState(x_init.copy(), hmc)
    hmc_samples = hmc.sample(n_samples)
    hmc_frames = [hmc_samples[:, f_idx * samples_per_frame].copy() for f_idx in xrange(0, n_frames)]

    # MJHMC
    print "MJHMC"
    mjhmc = MarkovJumpHMC(distribution=poe.reset(), resample=False, **mjhmc_params)
    mjhmc.state = HMCState(x_init.copy(), mjhmc)
    mjhmc_samples = mjhmc.sample(n_samples)
    mjhmc_frames = [mjhmc_samples[:, f_idx * samples_per_frame].copy() for f_idx in xrange(0, n_frames)]

    print mjhmc.r_count, hmc.r_count
    print mjhmc.l_count, hmc.l_count
    print mjhmc.f_count, hmc.f_count
    print mjhmc.fl_count, hmc.fl_count


    frames = [mjhmc_frames, hmc_frames]
    names = ['MJHMC', 'ControlHMC']
    frame_grads = [f_idx * samples_per_frame for f_idx in xrange(0, n_frames)]
    return frames, names, frame_grads


def plot_imgs(imgs, samp_names, step_nums, vmin = -2, vmax = 2):
    plt.figure(figsize=(5.5,3.6))

    nsamplers = len(samp_names)
    nsteps = len(step_nums)

    plt.subplot(nsamplers+1, nsteps+1, 1)
    plt.axis('off')
    plt.text(0.9, -0.1, "# grads",
        horizontalalignment='right',
        verticalalignment='bottom')

    for step_i in range(nsteps):
        plt.subplot(nsamplers+1, nsteps+1, 2 + step_i)
        plt.axis('off')
        plt.text(0.5, -0.1, "%d"%step_nums[step_i],
            horizontalalignment='center',
            verticalalignment='bottom')
    for samp_i in range(nsamplers):
        plt.subplot(nsamplers+1, nsteps+1, 1 + (samp_i+1)*(nsteps+1))
        plt.axis('off')
        plt.text(0.9, 0.5, samp_names[samp_i],
            horizontalalignment='right',
            verticalalignment='center')


    for samp_i in range(nsamplers):
        for step_i in range(nsteps):
            plt.subplot(nsamplers+1, nsteps+1, 2 + step_i + (samp_i+1)*(nsteps+1))

            ptch = imgs[samp_i][step_i].copy()
            img_w = np.sqrt(np.prod(ptch.shape))
            ptch = ptch.reshape((img_w, img_w))

            ptch -= vmin
            ptch /= vmax-vmin
            plt.imshow(ptch, interpolation='nearest', cmap=cm.Greys_r )
            plt.axis('off')

    # plt.tight_layout()
    plt.savefig('poe_samples.pdf')
    plt.close()



def plot_concat_imgs(imgs, border_thickness=2, axis=None, normalize=False):
    """ concatenate the imgs together into one big image separated by borders

    :param imgs: list or array of images. total number of images must be a perfect square and
      images must be square
    :param border_thickness: how many pixels of border between
    :param axis: optional matplotlib axis object to plot on
    :returns: array containing all receptive fields
    :rtype: array
    """
    sns.set_style('dark')
    assert isinstance(border_thickness, int)
    assert int(np.sqrt(len(imgs))) == np.sqrt(len(imgs))
    assert imgs[0].shape[0] == imgs[0].shape[1]
    if normalize:
        imgs = np.array(imgs)
        imgs /= np.sum(imgs ** 2, axis=(1,2)).reshape(-1, 1, 1)
    img_length = imgs[0].shape[0]
    layer_length = int(np.sqrt(len(imgs)))
    concat_length = layer_length * img_length + (layer_length - 1) * border_thickness
    border_color = np.nan
    concat_rf = np.ones((concat_length, concat_length)) * border_color
    for x_idx, y_idx in itertools.product(xrange(layer_length),
                                          xrange(layer_length)):
        # this keys into imgs
        flat_idx = x_idx * layer_length + y_idx
        x_offset = border_thickness * x_idx
        y_offset = border_thickness * y_idx
        # not sure how to do a continuation line cleanly here
        concat_rf[x_idx * img_length + x_offset: (x_idx + 1) * img_length + x_offset,
                  y_idx * img_length + y_offset: (y_idx + 1) * img_length + y_offset] = imgs[flat_idx]
    if axis is not None:
        axis.imshow(concat_rf, interpolation='none', aspect='auto')
    else:
        plt.imshow(concat_rf, interpolation='none', aspect='auto')

def ac_plot(n_samples=5000):
    """ Plots the autocorrelation for the best found parameters of the 36
    dimensional product of experts

    :returns: None
    :rtype: None
    """

    from mjhmc.figures.ac_fig import plot_best
    ndims = 36
    nbasis = 36

    np.random.seed(2015)
    poe = ProductOfT(nbatch=25,ndims=ndims,nbasis=nbasis)
    plot_best(poe, num_steps=n_samples, update_params=False)
