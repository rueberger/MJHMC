import numpy as np
from mjhmc.search.objective import obj_func
from mjhmc.samplers.markov_jump_hmc import MarkovJumpHMC
from mjhmc.misc.distributions import Gaussian

np.random.seed(2016)

def main(job_id, params):
    print "job id: {}, params: {}".format(job_id, params)
    return obj_func(MarkovJumpHMC, Gaussian(ndims=50, nbatch=50, log_conditioning=1.1),
                    job_id, **params)
