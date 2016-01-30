'''
Script to find best hyper params from a set of output logs

Inputs == directory of outputs (string)
Outputs == numpy array with values of min_objective, epsilon, beta, M

'''

import re
import glob
import numpy as np
import json


# format is [search_directory, num_hyperparameters ]
SEARCHES = [
    ["control_mm_gauss", 3],
    ["control_log_gauss", 3],
    ["control_rw", 3],
    ["MJHMC_mm_gauss", 3],
    ["MJHMC_log_gauss", 3],
    ["MJHMC_rw", 3],
    ["LAHMC_mm_gauss", 4],
    ["MJHMC_poe_36", 3],
    ["control_poe_36", 3],
    ["MJHMC_poe_100", 3],
    ["control_poe_100", 3]
]


def find(directory, num_params=3):
    """ finds all of the hyperparameters and their scores for a Spearmint search by scanning through
    the output logs

    no gaurantees if you run this on a debug log; clean out the output directory before starting
    new experiments

    :param directory: the directory of the output logs to scan through
    :param num_params: the number of hyperparameters. 3 for standard hmc, 4 for lahmc
    :returns: An array of valid hyperparameters and their scores
    :rtype: float32 array of shape (num_valid_parameters, num_parameters + 1)
            for lahmc, num_parameters = 4, for all other samplers num_parameters = 3

    """
    results = []
    files = glob.glob(directory + '*')
    if len(files) == 0:
        return None
    for fname in files:
        result = [None for _ in xrange(num_params + 1)]
        for line in open(fname,'r'):
            if re.search('u\'main\':',line):
                split_line = line.split('}]')
                if len(split_line) != 0:
                    uncast_param = split_line[0].split(':')[1]
                    result[0] = np.float32(uncast_param)
                else:
                    continue
            if re.search('epsilon',line):
                for idx in xrange(1, num_params + 1):
                    split_line = line.split('array([')
                    if len(split_line) != 0:
                        uncast_param = split_line[idx].split('])')[0]
                        result[idx] = np.float32(uncast_param)
                    else:
                        continue
            if len(result) == num_params + 1:
                results.append(result)

    if len(results) != 0:
        return np.array(results)

def write_best(results, directory):
    """ Finds the best in a set of a results, writes a json of the best
    parameters to file in directory

    :param results: array of results; the output of find()
    :param directory: directory of the search
    :returns: None, writes json to file
    :rtype: None
    """

    # occurs when search is running sometimes
    results = results[results[:, -1] != 0]
    min_idx = np.argmin(results[:, 0])
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
    """ Searches through all of output logs in the directories in SEARCHES
    if suitable output is found, params.json is written in the respective directory
    it contains the best hyperparameters found in the output logs

    :returns: None, writes params.json in search directories containing valid output logs
    :rtype: None
    """

    for directory, num_params in SEARCHES:

        results = find("{}/output/".format(directory), num_params)
        if results == None:
            print "No valid output found for {}".format(directory)
        else:
            write_best(results, directory)
            print "Parameters succesfully updated for {}".format(directory)
