from nuts import nuts6
import numpy as np
import pandas as pd

#average out gradient evaluations over samplings runs
# divide out by sampling step to get per sample grad evals, substract burn in period


def sample_nuts_to_df(distribution, n_samples=5000,
                      n_burn_in=5000, **kwargs):
    """
    runs NUTS on distribution for num_steps
    returns a dataframe containing samples, number of gradient evaluations at each step,
    distributions : initialized distributions object, must have nbatch=1
    sample_steps: the number of sampling steps concatenated into one time step. LAHMC used 10
    n_batch: the number of sampling runs from nuts concatenated into a batch
    n_burn_in: number of steps for NUTS to burn in and
    """
    assert distribution.nbatch == 1
    x_init = distribution.Xinit[:, 0]
    samples = nuts6(distribution, n_samples, n_burn_in, x_init)[0]
    grad_per_sample_step = distribution.dEdX_count / float(n_samples + n_burn_in)
    e_per_sample_step = distribution.E_count / float(n_samples + n_burn_in)
    print("grad per sample step: {}".format(grad_per_sample_step))

    recs = {}
    for t in range(n_samples):
        recs[t] = {
            'X': samples[t].reshape(distribution.ndims, 1),
            'num grad' : grad_per_sample_step * t,
            'num energy' : e_per_sample_step * t
        }
    df = pd.DataFrame.from_records(recs).T
    # might want to truncate df at gradient count
    return df
