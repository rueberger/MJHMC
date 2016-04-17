from mjhmc.search.objective import obj_func
from mjhmc.samplers.markov_jump_hmc import MarkovJumpHMC
from mjhmc.misc.distributions import ProductOfT
from scipy.sparse import rand
import numpy as np

np.random.seed(2015)


def main(job_id, params):
    ndims = 36
    nbasis = 36
    print "job id: {}, params: {}".format(job_id, params)
    return obj_func(MarkovJumpHMC, ProductOfT(nbatch=25,ndims=ndims,nbasis=nbasis), job_id, **params)
