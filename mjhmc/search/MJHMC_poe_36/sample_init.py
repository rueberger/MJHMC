'''
script that runs a really long chain and saves it as a pickle file
Caveats:
    This script needs to be run on whatever parameters we are going to do the search on
    So the parameters to the script are inputs that go into the search's parameters as
    well.
Outputs:
    It saves a pkl file in this directory.
    #TODO: This should become more general where you can set the distribution also as an input.
    For now, it will be a fixed thing
March 27th, 2016
'''

from mjhmc.samplers.markov_jump_hmc import MarkovJumpHMC
from mjhmc.misc.distributions import ProductOfT
from mjhmc.misc.autocor import sample_to_df
import numpy as np
import pickle
import IPython


if __name__=="__main__":
    ndims = 36
    nbasis = 36 
    num_steps = 10000
    print("Sampling to dataframe now....")
    df = sample_to_df(MarkovJumpHMC,ProductOfT(nbatch=100,ndims=ndims,nbasis=nbasis),num_steps=num_steps)
    '''
    pickle.dump(df['X'],open('poe_ndims_'+str(ndims) + '_nbasis_' + str(nbasis) \
            + '_nsamples_' + str(num_steps) +'.pkl','w'))
    '''
