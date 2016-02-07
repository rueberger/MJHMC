from mjhmc.search.objective import obj_func
from mjhmc.samplers.markov_jump_hmc import MarkovJumpHMC
from mjhmc.misc.distributions import ProductOfT
from scipy.sparse import rand
import numpy as np

np.random.seed(2015)


def main(job_id, params):
    ndims = 100 
    nbasis = 100 
    rand_val = rand(ndims,nbasis/2,density=0.25)
    print "job id: {}, params: {}".format(job_id, params)
    return obj_func(MarkovJumpHMC, ProductOfT(nbatch=25,ndims=ndims,nbasis=nbasis), job_id, **params)
