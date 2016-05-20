from mjhmc.search.objective import obj_func
from mjhmc.samplers.markov_jump_hmc import ControlHMC
from mjhmc.misc.distributions import ProductOfT
import numpy as np

np.random.seed(2015)


def main(job_id, params):
    ndims = 36
    nbasis = 36
    print "job id: {}, params: {}".format(job_id, params)
    return obj_func(ControlHMC, ProductOfT(nbatch=25, ndims=ndims, nbasis=nbasis), job_id, **params)
