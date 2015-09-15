from search.objective import obj_func
from samplers.markov_jump_hmc import MarkovJumpHMC
from misc.distributions import RoughWell


def main(job_id, params):
    print "job id: {}, params: {}".format(job_id, params)
    return obj_func(MarkovJumpHMC, RoughWell(nbatch=200), job_id, **params)
