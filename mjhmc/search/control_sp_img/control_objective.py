from mjhmc.search.objective import obj_func
from mjhmc.samplers.markov_jump_hmc import ControlHMC
from mjhmc.misc.tf_distributions import SparseImageCode

def main(job_id, params):
    print "job id: {}, params: {}".format(job_id, params)
    return obj_func(ControlHMC, SparseImageCode(n_patches=9, n_batches=10,
                                                device='/gpu:0', gpu_frac=0.5), job_id, **params)
