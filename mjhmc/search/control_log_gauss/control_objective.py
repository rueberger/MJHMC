from search.objective import obj_func
from samplers.markov_jump_hmc import ControlHMC
from misc.distributions import Gaussian


def main(job_id, params):
    print "job id: {}, params: {}".format(job_id, params)
    return obj_func(ControlHMC, Gaussian(ndims=10, nbatch=200), job_id, **params)