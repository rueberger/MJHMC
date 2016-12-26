from LAHMC import LAHMC
from mjhmc.search.objective import obj_func
from mjhmc.misc.distributions import MultimodalGaussian


def main(job_id, params):
    params.update({"display": [0]})
    print("job id: {}, params: {}".format(job_id, params))
    return obj_func(LAHMC, MultimodalGaussian(ndims=5, separation=1),
                    job_id,
                    **params)
