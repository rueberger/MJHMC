"""
This module contains a script for generating the product of experts image patch figure
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy.sparse import rand

from mjhmc.misc.distributions import ProductOfT

plt.ion()


# for deterministic params for poet
np.random.seed(2015)

def generate_figure(samplers, samples_per_frame=100, n_frames=None):
    """ Generates the figure

    :param samplers: list of samplers to run on
    :param samples_per_frame: number of sample steps between each frame
    :param n_frames: number of frames to draw
    :returns: None
    :rtype: None
    """
    # will need to set distribution x_init so that all start in same state
    n_frames = n_frames or len(samplers)
    ndims = 36
    nbasis = 72
    rand_val = rand(ndims,nbasis/2,density=0.25)
    W = np.concatenate([rand_val.toarray(), -rand_val.toarray()],axis=1)
    logalpha = np.random.randn(nbasis, 1)

    poe = ProductOfT(nbatch=1, W=W, logalpha=logalpha)
    frames = []
    for _, sampler in enumerate(samplers):
        samples = sampler(distribution=poe.reset()).sample(samples_per_frame * n_frames)
        for f_idx in xrange(n_frames):
            s_idx = f_idx * samples_per_frame
            frames.append(samples[:, s_idx].reshape(np.sqrt(poe.ndims), np.sqrt(poe.ndims)))
    plot_concat_imgs(frames)


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