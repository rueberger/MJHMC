"""
This module contains utilities for computing the autocorrelation of a sequence of samples
"""
import pandas as pd
from mklfft.fftpack import fftn, ifftn
import numpy as np
from time import time



def calculate_autocorrelation(sampler, distribution,
                              num_steps=None, num_grad_steps=None,
                              sample_steps=1, half_window=False,
                              use_cached_var=False,**kwargs):
    """
    just a helper function
    refer to the docstrings for the respective methods
    """
    print "Now generating samples..."
    start_time = time()
    samples, e_evals, grad_evals = generate_samples(sampler, distribution.reset(),
                                                    num_steps, num_grad_steps, **kwargs)
    print "Took {} seconds".format(time() - start_time)

    cached_var = None
    if use_cached_var:
        print "Using cached variance"
        _, emc_var_estimate, true_var_estimate = distribution.load_cache()
        if sampler.__name__ == "MarkovJumpHMC":
            cached_var = emc_var_estimate
        else:
            cached_var = true_var_estimate

    print "Calculating autocorrelation..."
    return autocorrelation(samples, e_evals, grad_evals, half_window, cached_var=cached_var)

def fft_autocor(samples):
    """ Calculate autocorrelation using the cross-correlation theorem

    Args:
      samples: array of samples - [n_dims, n_batch, n_samples]

    Returns:
       autocor: [n_samples]
    """
    assert samples.ndim == 3
    fft_samples = fftn(samples, axes=[-1])
    return np.real(np.mean(ifftn(fft_samples * np.conj(fft_samples), axes=[-1]), axis=(0, 1)))


def autocorrelation(samples, e_evals, grad_evals, half_window=True,
                    normalize=True, cached_var=None, brute_force=False,
                    use_tf=False):
    n_dims, n_batch, n_samples = samples.shape

    if brute_force:
        if use_tf:
            import tensorflow as tf
            with tf.Graph().as_default(), tf.Session() as sess:
                print "Building autocor op"
                ac_op, samples_pl = build_autocor_op(n_dims, n_batch, n_samples, half_window=half_window)

                print "Initializing variables"
                sess.run(tf.initialize_all_variables())

                print "Calculating autocor"
                ac_squeeze = sess.run(ac_op, feed_dict={samples_pl: samples})
        else:
            import theano
            import theano.tensor as T
            print "Now compiling autocorrelation function"
            start_time = time()
            theano_ac = compile_autocor_func(half_window)
            print "Took {} seconds".format(time() - start_time)

            print "Now running compiled autocorrelation function..."
            start_time = time()
            raw_autocor = theano_ac(samples.astype('float32'))
            print "Took {} seconds".format(time() - start_time)
            ac_squeeze = np.squeeze(raw_autocor[0])
    else:
        ac_squeeze = fft_autocor(samples)

    if cached_var is None:
        # variance given assumption of *zero mean*
        print "Computing sample variance with numpy..."
        var = np.mean(samples**2, keepdims=True)[0][0]
    else:
        var = cached_var

    print "Now moving result to dataframe"
    start_time = time()

    if normalize:
        ac_squeeze = ac_squeeze / var
        # theano doesn't play nice with the first element but it's just the variance
        autocor = np.vstack((1., ac_squeeze.reshape(-1, 1)))
    else:
       # theano doesn't play nice with the first element but it's just the variance
        autocor = np.vstack((var, ac_squeeze.reshape(-1, 1)))


    #This drops the last sample. Unclear, if this is the best way to do things but
    #it is the only way we can align the total number of samples from sample generation to
    #computing autocorrelation
    if half_window:
        e_evals = e_evals[:int(n_samples / 2) - 1]
        grad_evals = grad_evals[:int(n_samples / 2) - 1]

    else:
        e_evals = e_evals[:-1]
        grad_evals = grad_evals[:-1]
    print "Took {} seconds".format(time() - start_time)
    return autocor, e_evals, grad_evals



def compile_autocor_func(half_window):
    X = T.tensor3().astype('float32')
    shape = X.shape
    #Assumes Length T, need to have a switch that also deals with (T/2)-1
    if half_window:
        t_gap = T.arange(1, (shape[2] / 2) - 1)
    else:
        t_gap = T.arange(1, shape[2] - 1)
    outputs_info = T.zeros((1, 1, 1), dtype='float32')

    #function def that computes the mean ac for each time lag
    def calc_ac(t_gap, _, X):
        return T.mean(X[:, :, :-t_gap] * X[:, :, t_gap:], dtype='float32', keepdims=True)

    result, updates = theano.scan(fn=calc_ac,
                                 outputs_info=[outputs_info],
                                 sequences=[t_gap],
                                 non_sequences=[X])
    theano_ac = theano.function(inputs=[X], outputs=[result], updates=updates)
    return theano_ac

def build_autocor_op(n_dims, n_batch, n_samples, half_window=True):
    """ Builds a tensorflow op to comptue the autocorrelation of a sequence of samples

    Args:
       half_window: If true, computes autocorrelation out to half of the length of the
          input sequence, to reduce noise
       n_dims: int
       n_batch: int
       n_samples: int

    Returns:
       autocorrelation: tensor - [max_t]
       samples_pl: placeholder for samples - [n_dims, n_batch, n_samples]
    """
    import tensorflow as tf
    if half_window:
        max_t = (n_samples / 2) - 1
    else:
        max_t = n_samples - 1

    samples_pl = tf.placeholder(tf.float32, shape=(n_dims, n_batch, n_samples), name='samples_pl')

    def ac_elem(t_idx):
        tf.reduce_mean(samples_pl[:, :, :n_samples - t_idx] * samples_pl[:, :, t_idx:])

    autocor = tf.map_fn(ac_elem, tf.constant(np.arange(1, max_t)),
                        dtype=tf.float32,
                        parallel_iterations=50,
                        back_prop=False,
                        swap_memory=False)


    return autocor, samples_pl


def slow_autocorrelation(samples, e_evals, grad_evals, half_window=False):
    """
    calculate the autocorrelation
    history a dataframe
    assumes zero mean
    history is a data frame

    returns the dataframe: autocorrelation gradient evaluations energy evaluations
    """

    N, nbatch, T = samples.shape

    if not half_window:
        c = np.zeros((T-1,))
        # variance given assumption of zero mean
        c[0] = np.mean(samples**2)
        for t_gap in range(1, T-1):
            c[t_gap] = np.mean(samples[:,:,:-t_gap]*samples[:,:,t_gap:])

        # can't have full length window
        # not sure that truncating at the end is the proper thing to do
        autocor = c / c[0]

    if half_window:
        c = np.zeros(((T/2)-1,))
        # variance given assumption of zero mean
        c[0] = np.mean(samples**2)
        for t_gap in range(1, (T/2)-1):
            c[t_gap] = np.mean(samples[:,:,:-t_gap]*samples[:,:,t_gap:])

        # can't have full length window
        # not sure that truncating at the end is the proper thing to do
        autocor = c / c[0]

    return autocor, e_evals, grad_evals

def generate_samples(sampler, distribution, num_steps=None, num_grad_steps=None,
                     **kwargs):
    """ Generate samples *without* using a dataframe

    Args:
       sampler: sampler class
       distribution: distribution object
       num_steps: number of desired steps - optional
       num_grad_steps: number of desired grad steps - optional

    Returns:
       (samples - [n_dims, n_batch, n_samples]
        e_evals - [n_samples]
        grad_evals - [n_samples])
    """
    # ridiculous assert to make sure only one of them is ever None
    assert (((num_steps is None) and (num_grad_steps is not None)) or
            (num_steps is not None) and (num_grad_steps is None))
    smp = sampler(distribution=distribution, **kwargs)
    # fudge factor because grad per sampler step is only approximate
    num_steps = num_steps or num_grad_steps / smp.grad_per_sample_step + 100

    n_dims = distribution.ndims
    n_batch = distribution.nbatch
    # [n_dims, n_batch, num_steps]
    samples = np.zeros((n_dims, n_batch, num_steps))

    grad_evals = np.zeros(num_steps)
    e_evals = np.zeros(num_steps)

    # reset counters
    distribution.reset()
    for t_idx in range(num_steps):
        samples[:, :, t_idx] = sampler.sample(1)
        grad_evals[t_idx] = distribution.dEdX_count / float(n_batch)
        e_evals[t_idx] = distribution.E_count / float(n_batch)

    if num_grad_steps is not None:
        assert grad_evals[-1] >= num_grad_steps
        grad_sel = (grad_evals <= num_grad_steps)
        return samples[:, :, grad_sel], e_evals[grad_sel], grad_evals[grad_sel]
    else:
        return samples, e_evals, grad_evals


def sample_to_df(sampler, distribution, num_steps=None, num_grad_steps=None,
                 sample_steps=1, **kwargs):
    """
    runs the specified sampler on the distribution
    returns a dataframe containing samples, number of gradient evaluations at each step,
    and the energy evaluations
    distributions : initialized distributions object
    sample_steps: the number of sampling steps concatenated into one time step. LAHMC used 10
    num_steps: number of sampling steps
    num_grad_steps: number of target grad steps, can either specify steps or grads, not both
    """
    # ridiculous assert to make sure only one of them is ever None
    assert (((num_steps is None) and (num_grad_steps is not None)) or
            (num_steps is not None) and (num_grad_steps is None))
    smp = sampler(distribution=distribution, **kwargs)
    # fudge factor because grad per sampler step is only approximate
    num_steps = num_steps or num_grad_steps / smp.grad_per_sample_step + 100
    # {time : {'X': samples, 'num grad' dEdX evals, 'num energy': E evals}}
    recs = {}
    smp.burn_in()
    distribution.reset()
    for t in xrange(num_steps):
        recs[t] = {
            'X': smp.sample(sample_steps),
            # cumulative
            'num grad': distribution.dEdX_count / distribution.nbatch,
            'num energy': distribution.E_count / distribution.nbatch,
            # change this to a fraction later
            # 'tot L': smp.L_count,
            # 'tot FL': smp.FL_count,
            # 'tot F': smp.F_count,
            # 'tot R': smp.R_count
        }
    df = pd.DataFrame.from_records(recs).T
    if num_grad_steps is not None:
        assert df.iloc[-1]['num grad'] >= num_grad_steps
        truncated_df = df.loc[:, 'num grad'] >= num_grad_steps
        trunc_idx = truncated_df[truncated_df].index[0]
        return smp, df.loc[:trunc_idx]
    else:
        return smp, df
