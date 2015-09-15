from search.objective import obj_func
from LAHMC import LAHMC
from misc.distributions import RoughWell


def main(job_id, params):
    params.update({"display": [0]})
    print "job id: {}, params: {}".format(job_id, params)
    return obj_func(LAHMC, RoughWell(nbatch=100),
                    job_id,
                    **params)
