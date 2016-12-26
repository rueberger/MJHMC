from mjhmc.search.objective import obj_func
from mjhmc.samplers.markov_jump_hmc import MarkovJumpHMC
from mjhmc.misc.distributions import ProductOfT
from scipy.sparse import rand
import numpy as np

np.random.seed(2015)


def main(job_id, params):
    ndims = 256
    nbasis = 72
    rand_val = rand(ndims,nbasis/2,density=0.25)
    W = np.concatenate([rand_val.toarray(), -rand_val.toarray()],axis=1)
    logalpha = np.random.randn(nbasis, 1)
    print("job id: {}, params: {}".format(job_id, params))
    return obj_func(MarkovJumpHMC, ProductOfT(nbatch=250,ndims=ndims,nbasis=nbasis, W=W, logalpha=logalpha), job_id, **params)
