"""
This module contains utilities for computing the autocorrelation of a sequence of samples
"""
import pandas as pd
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
    smp, sample_df = sample_to_df(sampler, distribution.reset(), num_steps, num_grad_steps,
                      sample_steps, **kwargs)
    print "Took {} seconds".format(time() - start_time)

    cached_var = None
    if use_cached_var:
        print "Using cached variance"
        _, emc_var_estimate, true_var_estimate = distribution.load_cache()
        if smp.distribution.mjhmc:
            cached_var = emc_var_estimate
        else:
            cached_var = true_var_estimate

    print "Calculating autocorrelation..."
    return autocorrelation(sample_df, half_window, cached_var=cached_var)

def autocorrelation(history, half_window=True, normalize=True, cached_var=None, use_tf=False):
    n_samples = len(history)
    n_dims, n_batch = history.loc[0]['X'].shape

    samples = np.zeros((n_dims, n_batch, n_samples))

    print "Copying samples to array"
    start_time = time()
    for idx in range(n_samples):
        samples[:, :, idx] = history.loc[idx]['X']
    print "Took {} seconds".format(time() - start_time)


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



    #This drops the last sample out of the data frame. Unclear, if this is the best way to do things but
    #it is the only way we can align the total number of samples from sample generation to
    #computing autocorrelation
    if half_window:
        ac_df = history[:int(n_samples / 2) - 1]
    else:
        ac_df = history[:-1]
    ac_df.loc[:, 'autocorrelation'] = autocor
    print "Took {} seconds".format(time() - start_time)
    return ac_df[['num energy', 'num grad', 'autocorrelation']]



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


def slow_autocorrelation(history, half_window=False):
    """
    calculate the autocorrelation
    history a dataframe
    assumes zero mean
    history is a data frame

    returns the dataframe: autocorrelation gradient evaluations energy evaluations
    """
    # number of steps
    T = len(history)
    # distribution dimensions, batch size
    N, nbatch = history.loc[0]['X'].shape

    X = np.zeros((N, nbatch, T))
    for tt in range(T):
        X[:, :, tt] = history.loc[tt]['X']

    if not half_window:
        c = np.zeros((T-1,))
        # variance given assumption of zero mean
        c[0] = np.mean(X**2)
        for t_gap in range(1, T-1):
            c[t_gap] = np.mean(X[:,:,:-t_gap]*X[:,:,t_gap:])

        # can't have full length window
        # not sure that truncating at the end is the proper thing to do
        ac_df = history[:-1]
        ac_df.loc[:, 'autocorrelation'] = c/c[0]

    if half_window:
        c = np.zeros(((T/2)-1,))
        # variance given assumption of zero mean
        c[0] = np.mean(X**2)
        for t_gap in range(1, (T/2)-1):
            c[t_gap] = np.mean(X[:,:,:-t_gap]*X[:,:,t_gap:])

        # can't have full length window
        # not sure that truncating at the end is the proper thing to do
        ac_df = history[:(T/2)-1]
        ac_df.loc[:, 'autocorrelation'] = c/c[0]

    return ac_df[['num energy', 'num grad', 'autocorrelation']]


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
