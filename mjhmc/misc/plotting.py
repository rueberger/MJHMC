"""
This file contains various plotting utilities
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


from mjhmc.samplers.markov_jump_hmc import HMCBase, ContinuousTimeHMC, MarkovJumpHMC
from mjhmc.misc.distributions import TestGaussian
from .autocor import calculate_autocorrelation

import numpy as np

# some formatting for seaborn
sns.set_palette("deep", desat=.6)
sns.set_context(rc={"figure.figsize": (8, 4)})

#pylint: disable=too-many-arguments

def plot_search_ac(t, ac, job_id, params, score):
    plt.plot(t, ac)
    plt.title("Score: {}, beta: {}, epsilon: {}, M: {}".format(score,
        params['beta'], params['epsilon'], params['num_leapfrog_steps']))
    plt.savefig("job_{}_ac.png".format(job_id))

def plot_fit(grad_evals, autocor, exp_coef, cos_coef, job_id, params, save=True):
    """ Debug plot for hyperparameter search
    Saves a plot of the autocorrelation and the fitted complex exponential of the form
    ac(t) = exp(a * t) * cos(b * t)

    :param grad_evals: array of grad evals. Use as x-axis
    :param autocor: array of autocor. Of same shape as grad_evals
    :param exp_coef: float. Parameter a for fitted func
    :param cos_coef: float. Parameter b for fitted func
    :param job_id: the id of the Spearmint job that tested these params
    :param params: the set of parameters for this job
    :returns: the rendered figure
    :rtype: plt.figure()
    """
    fig = plt.figure()
    plt.plot(grad_evals, autocor, label='observed')
    fitted = np.exp(exp_coef * grad_evals) * np.cos(cos_coef * grad_evals)
    plt.plot(grad_evals, fitted, label="fittted")
    plt.title("Score: {}, beta: {}, epsilon: {}, M: {}".format(
        exp_coef, params['beta'], params['epsilon'], params['num_leapfrog_steps']))
    plt.legend()
    if save:
        plt.savefig("job_{}_fit.pdf".format(job_id))
    return fig


def hist_1d(distr, nsamples=1000, nbins=250, control=True, resample=True):
    """
    plots a 1d histogram from each sampler
    distr is (an unitialized) class from distributions
    """
    distribution = distr(ndims=1)
    control_smp = HMCBase(distribution=distribution, epsilon=1)
    experimental_smp = MarkovJumpHMC(distribution=distribution, resample=resample, epsilon=1)

    if control:
        plt.hist(control_smp.sample(nsamples)[0], nbins, normed=True, label="Standard HMCBase", alpha=.5)

    plt.hist(experimental_smp.sample(nsamples)[0], nbins, normed=True, label="Continuous-time HMCBase",alpha=.5)
    plt.legend()

def gauss_1d(nsamples=1000, nbins=250, *args, **kwargs):
    """
    Simple test plots.
    Draws nsamples with sampler from a (well conditioned unit variance)
    gaussian and plots a histogram of both of them
    """
    hist_1d(TestGaussian, nsamples, nbins, *args, **kwargs)
    test_points = np.linspace(4, -4, 1000)
    true_curve = (1. / np.sqrt(2*np.pi)) * np.exp(- (test_points**2) / 2.)
    plt.plot(test_points, true_curve, label="True curve")
    plt.legend()


def gauss_2d(nsamples=1000):
    """
    Another simple test plot
    1d gaussian sampled from each sampler visualized as a joint 2d gaussian
    """
    gaussian = TestGaussian(ndims=1)
    control = HMCBase(distribution=gaussian)
    experimental = MarkovJumpHMC(distribution=gaussian, resample=False)


    with sns.axes_style("white"):
        sns.jointplot(
            control.sample(nsamples)[0],
            experimental.sample(nsamples)[0],
            kind='hex',
            stat_func=None)

def hist_2d(distr, nsamples, **kwargs):
    """
    Plots a 2d hexbinned histogram of distribution

    Args:
     distr: Distribution object
     nsamples: number of samples to use to generate plot
    """
    sampler = MarkovJumpHMC(distribution=distr, **kwargs)
    samples = sampler.sample(nsamples)

    with sns.axes_style("white"):
       g =  sns.jointplot(samples[0], samples[1], kind='kde', stat_func=None)
    return g

def jump_plot(distribution, nsamples=100, **kwargs):
    """
    Plots samples drawn from distribution with dwelling time on the x-axis
    and the sample value on the y-axis
    1D only
    """
    distr = distribution(ndims=1, nbatch=1, **kwargs)
    # sampler = MarkovJumpHMC(np.array([0]).reshape(1,1),
    sampler = MarkovJumpHMC(distr.Xinit,
                            distr.E, distr.dEdX,
                            epsilon=.3, beta=.2, num_leapfrog_steps=5)
    x_t = []
    d_t = []
    transitions = []
    last_L_count, last_F_count, last_R_count = 0, 0, 0
    for idx in xrange(nsamples):
        sampler.sampling_iteration()
        x_t.append(sampler.state.X[0, 0])
        d_t.append(sampler.dwelling_times[0])
        if sampler.L_count - last_L_count == 1:
            transitions.append("L")
            last_L_count += 1
        elif sampler.F_count - last_F_count == 1:
            transitions.append("F")
            last_F_count += 1
        elif sampler.R_count - last_R_count == 1:
            transitions.append("R")
            last_R_count += 1
    t = np.cumsum(d_t)
    plt.scatter(t, x_t)
    t = np.array(t).reshape(len(t), 1)
    x_t = np.array(x_t).reshape(len(x_t), 1)
    transitions = np.array(transitions).reshape(len(transitions), 1)
    data = np.concatenate((x_t, t, transitions), axis=1)
    return pd.DataFrame(data, columns=['x', 't', 'transitions'])



def plot_autocorrelation(distribution, ndims=2,
                         num_steps=200, nbatch=100, sample_steps=10):
    """
    plot autocorrelation versus gradient evaluations

    set to be deprecated
    """
    # TODO: bring up to speed of DF-less calc autocor
    distr = distribution(ndims, nbatch)
    d_ac = calculate_autocorrelation(HMCBase, distr, num_steps, sample_steps)
    c_ac = calculate_autocorrelation(ContinuousTimeHMC, distr, num_steps, sample_steps)
    d_ac.index = d_ac['num grad']
    c_ac.index = c_ac['num grad']
    # combine plots, maybe join possible after all
    d_ac['autocorrelation'].plot(label='discrete')
    c_ac['autocorrelation'].plot(label='continuous')
    plt.legend()


def scale_to_unit_interval(ndar, eps=1e-8):
      """ Scales all values in the ndarray ndar to be between 0 and 1 """
      ndar = ndar.copy()
      ndar -= ndar.min()
      ndar *= 1.0 / (ndar.max() + eps)
      return ndar

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """ Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape = [0,0]
    # out_shape[0] = (img_shape[0] + tile_spacing[0]) * tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1] + tile_spacing[1]) * tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                 in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                out_array[:, :, i] = np.zeros(out_shape,
                                                 dtype='uint8' if output_pixel_vals else out_array.dtype
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing, scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        out_array = np.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)


        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                    else:
                        this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    out_array[
                        tile_row * (H+Hs): tile_row * (H + Hs) + H,
                        tile_col * (W+Ws): tile_col * (W + Ws) + W

                    ] \
                    = this_img * (255 if output_pixel_vals else 1)
        return out_array
