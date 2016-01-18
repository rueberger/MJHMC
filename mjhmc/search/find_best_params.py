'''
Script to find best hyper params from a set of output logs

Inputs == directory of outputs (string)
Outputs == numpy array with values of min_objective, epsilon, beta, M

'''

import re
import glob
import numpy as np
import json


searches = [
    ["control_mm_gauss", True],
    ["control_log_gauss", True],
    ["control_rw", True],
    ["MJHMC_mm_gauss", True],
    ["MJHMC_log_gauss", True],
    ["MJHMC_rw", True],
    ["LAHMC_mm_gauss", True],
    ["MJHMC_poe_36", True],
    ["control_poe_36", True]
]


def find(directory, lahmc=False):
    """ finds all of the hyperparameters and their scores for a Spearmint search by scanning
    through the output logs

    :param directory: the directory of the output logs to scan through
    :param lahmc: boolean flag, true for lahmc
    :returns: An array of valid hyperparameters and their scores
    :rtype: float32 array of shape (num_valid_parameters, num_parameters + 1)
            for lahmc, num_parameters = 4, for all other samplers num_parameters = 3
    """
    #List files
    files = glob.glob(directory + '*')
    if len(files) == 0:
        return None
    #Result array
    if lahmc:
        results = np.zeros((len(files),5),dtype='float32')
    else:
        results = np.zeros((len(files),4),dtype='float32')
    for idx, fname in enumerate(files):
        for line in open(fname,'r'):
            if re.search('u\'main\':',line):
                split_line = line.split('}]')
                if len(split_line) != 0:
                    uncast_param = split_line[0].split(':')[1]
                    results[idx, 0] = np.float32(uncast_param)
                else:
                    continue
            if re.search('epsilon',line):
                for param_idx in xrange(4):
                    split_line = line.split('array([')
                    if len(split_line) != 0:
                        uncast_param = split_line[1].split('])')[0]
                        results[idx, param_idx] = np.float32(uncast_param)
                    else:
                        continue
                if lahmc:
                    results[idx, 4] = np.float32(line.split('\'num_look_ahead_steps\': ')[1].split(',')[0])
    return results

def write_best(results, directory):
    # occurs when search is running sometimes
    results = results[results[:, -1] != 0]
    min_idx = np.argmin(results[:,0])
#    min_idx = np.argsort(results[:,0])[1]
    best_par = results[min_idx]
    params = {
        'epsilon' : float(best_par[1]),
        'beta' : float(best_par[2]),
        'num_leapfrog_steps' : int(best_par[3])
    }
    if len(best_par) == 5:
        params.update({'num_look_ahead_steps' : int(best_par[4])})
    with open("{}/params.json".format(directory), 'w') as d:
        json.dump(params, d)

def write_all():
    for directory, flag in searches:
        if flag == True:
            results = find("{}/output/".format(directory), lahmc=False)
            if results == None:
                print "No output found for {}".format(directory)
            else:
                write_best(results, directory)
                print "Parameters succesfully updated for {}".format(directory)
