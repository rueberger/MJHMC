import numpy as np
from search.objective import obj_func
from samplers.markov_jump_hmc import ControlHMC
from misc.distributions import Gaussian

np.random.seed(2016)

def main(job_id, params):
    print "job id: {}, params: {}".format(job_id, params)
    return obj_func(ControlHMC, Gaussian(ndims=50, nbatch=50, log_conditioning=1.1),
                    job_id, **params)
