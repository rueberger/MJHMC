from mjhmc.search.objective import obj_func
from mjhmc.samplers.markov_jump_hmc import MarkovJumpHMC
from mjhmc.misc.distributions import ProductOfT
from scipy.sparse import rand
import numpy as np


def main(job_id, params):
    ndims = 36
    nbasis = 36
    print("job id: {}, params: {}".format(job_id, params))
    weights, lognu = init_weights(ndims, nbasis)
    return obj_func(MarkovJumpHMC, ProductOfT(nbatch=25, ndims=ndims, nbasis=nbasis, lognu=lognu, W=weights), job_id, **params)

def init_weights(ndims, nbasis):
    """ Sparse normal weights,
    """
    np.random.seed(2015)
    sp_var = np.random.rand(ndims, nbasis)
    w_sp = np.random.randn(ndims, nbasis)
    w_sp[sp_var > 0.05] = 0
    lognu = np.log(np.random.rand(nbasis,) * 2 + 2.1)
    return w_sp, lognu
