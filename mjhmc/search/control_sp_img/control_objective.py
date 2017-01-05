from mjhmc.search.objective import obj_func
from mjhmc.samplers.markov_jump_hmc import ControlHMC
from mjhmc.misc.tf_distributions import SparseImageCode

def main(job_id, params):
    print "job id: {}, params: {}".format(job_id, params)
    # counter-intuitively, benchmarks indicate that this is optimal
    device_dict = {'grad': '/cpu:0',
                   'energy': '/gpu:0'}
    return obj_func(ControlHMC, SparseImageCode(n_patches=9, n_batches=10,
                                                device=device_dict, gpu_frac=1), job_id, **params)
