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

from mjhmc.samplers.markov_jump_hmc import ControlHMC
from mjhmc.misc.distributions import ProductOfT
from mjhmc.misc.autocor import sample_to_df
import numpy as np
import pickle
import IPython


if __name__=="__main__":
    ndims = 36
    nbasis = 36 
    num_steps = 10000
    np.random.seed(12345)
    print("Sampling to dataframe now....")
    IPython.embed()
    df = sample_to_df(ControlHMC,ProductOfT(nbatch=100,ndims=ndims,nbasis=nbasis),num_steps=num_steps)
    last_sample = df['X'].as_matrix()[-1]
    print last_sample
    pickle.dump(last_sample,open('poe_ndims_'+str(ndims) + '_nbasis_' + str(nbasis) \
            + '_nsamples_' + str(num_steps) +'.pkl','w'))
   
