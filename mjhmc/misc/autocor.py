"""
This module contains utilities for computing the autocorrelation of a sequence of samples
"""
import pandas as pd
import numpy as np
import theano
import theano.tensor as T


def calculate_autocorrelation(sampler, distribution,
                              num_steps=None, num_grad_steps=None,
                              sample_steps=1, half_window=False,
                              **kwargs):
    """
    just a helper function
    refer to the docstrings for the respective methods
    """
    df = sample_to_df(sampler, distribution.reset(), num_steps, num_grad_steps,
                      sample_steps, **kwargs)
    # tot_samples = float(df.iloc[-1]['tot L'] + df.iloc[-1]['tot F']
    #                     + df.iloc[-1]['tot R'] + df.iloc[-1]['tot FL'])
    # print "Sampler: {}; transitions: L: {}, F: {}, R: {}, FL: {}".format(
    #     sampler.__name__,
    #     df.iloc[-1]["tot L"] / tot_samples,
    #     df.iloc[-1]["tot F"] / tot_samples,
    #     df.iloc[-1]['tot R'] / tot_samples,
    #     df.iloc[-1]['tot FL'] / tot_samples
    # )
    return autocorrelation(df, half_window)

def autocorrelation(history, half_window=False, normalize=True):

    theano_ac = compile_autocor_func(half_window)

    n_samples = len(history)
    n_dims, n_batch = history.loc[0]['X'].shape

    samples = np.zeros((n_dims, n_batch, n_samples))

    for idx in range(n_samples):
        samples[:, :, idx] = history.loc[idx]['X']

    raw_autocor = theano_ac(samples.astype('float32'))

    # variance given assumption of *zero mean*
    sample_var = np.mean(samples**2, keepdims=True)[0][0]
    ac_squeeze = np.squeeze(raw_autocor[0])
    if normalize:
        ac_squeeze = ac_squeeze / sample_var
        # theano doesn't play nice with the first element but it's just the variance
        autocor = np.vstack((1, ac_squeeze.reshape(n_samples-2, 1)))
    else:
       # theano doesn't play nice with the first element but it's just the variance
        autocor = np.vstack((sample_var, ac_squeeze.reshape(n_samples-2, 1)))

    #This drops the last sample out of the data frame. Unclear, if this is the best way to do things but
    #it is the only way we can align the total number of samples from sample generation to
    #computing autocorrelation
    ac_df = history[:-1]
    ac_df.loc[:, 'autocorrelation'] = autocor
    return ac_df[['num energy', 'num grad', 'autocorrelation']]



def compile_autocor_func(half_window):
    X = T.tensor3().astype('float32')
    shape = X.shape
    #Assumes Length T, need to have a switch that also deals with (T/2)-1
    if half_window:
        t_gap = T.arange(1, (shape[2] / 2) - 1)
    else:
        t_gap = T.arange(1, shape[2] - 1)
    outputs_info = T.zeros((1,1,1),dtype='float32')

    #function def that computes the mean ac for each time lag
    def calc_ac(t_gap, _, X):
        return T.mean(X[:,:,:-t_gap]*X[:,:,t_gap:],dtype='float32',keepdims=True)

    #We will write a scan function that loops over the indices of the data tensor
    #and computes the autocorrelation
    result,updates = theano.scan(fn= calc_ac,
    #ac,updates = theano.scan(fn= lambda X,t_gap,ac: T.mean(X[:,:,:-t_gap]*X[:,:t_gap:]),
            outputs_info=[outputs_info],
            sequences=[t_gap],
            non_sequences=[X])
    #Append zero mean value of X to the front of the array and then return
    #Also, need to divide by the first element to scale the variances
    #For now though, let's do this in the main script
    theano_ac = theano.function(inputs=[X],outputs=[result],updates=updates)
    return theano_ac



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
    smp = sampler(distribution.Xinit, distribution.E, distribution.dEdX, **kwargs)
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
        return df.loc[:trunc_idx]
    else:
        return df
