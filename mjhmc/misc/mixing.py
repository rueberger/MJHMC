import numpy as np
from scipy.linalg import eig
from mjhmc.samplers.algebraic_hmc import AlgebraicDiscrete, AlgebraicContinuous
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from exceptions import RuntimeError

def get_eigs(sampler, order, steps=1000, energies=None):
    """Runs the sampler, returns the l1 normalized eigs
    """
    hmc = sampler(order, energies=energies)
    for _ in xrange(steps):
        hmc.sampling_iteration()
    t = hmc.get_transition_matrix()
    return eig(t, left=True, right=False)


def mixing_times(H, trials=10):
    """runs the two samplers with the given energy a bunch of times
       reports back their average mixing times
    """
    order = len(H) * 2
    c_tm = np.zeros(trials)
    d_tm = np.zeros(trials)

    for i in xrange(trials):
        # todo: add reset methods
        print "trial: {}".format(i)
        hmc = AlgebraicDiscrete(order, energies=H)
        chmc = AlgebraicContinuous(order, energies=H)
        d_tm[i] = hmc.calculate_mixing_time()
        c_tm[i] = chmc.calculate_mixing_time()

    print "Average mixing time for discrete sampler: {}".format(np.mean(d_tm))
    print "Average mixing time for continuous sampler: {}".format(np.mean(c_tm))

def test_sampler(sampler, H, steps=1000):
    """Runs the sampler on the given energy
    Prints a bunch of statistics about how well it's doing
    returns t_obs, distr_obs
    """
    order = len(H) * 2
    smp = sampler(order, energies=H)

    smp.sample(steps)
    t_obs = smp.get_transition_matrix()

    print "Predicted distribution: {}  \n".format(smp.prd_distr)
    print "Observed distribution: {} \n".format(smp.get_distr())
    print "Sampling error (L1): {} \n".format(smp.sampling_err())
    print "Observed transition matrix: \n {} \n".format(t_obs)
    print "Eigenspectrum of observed transition matrix: \n"
    eigs = rectify_evecs(eig(t_obs, left=True, right=False))
    pprint_eigs(eigs)

    return t_obs, smp.get_distr()

def pprint_eigs(eigs):
    """eigs: output of linalg.eig
    pretty prints the results
    """
    for l, vec in zip(eigs[0], eigs[1]):
        print "Eigenvalue: {} \n".format(l)
        print "Eigenvector: {} \n".format(list(vec))

def rectify_evecs(eigs):
    """
    eigs: output of linalg.eig
    normalizes evecs by L1 norm, truncates small complex components,
    ensures things are positive
    """
    evecs = eigs[1].T
    l1_norm = np.abs(evecs).sum(axis=1)
    norm_evecs = evecs / l1_norm[:, np.newaxis]
    real_evals = [np.around(np.real_if_close(l), decimals=5) for l in eigs[0]]
    real_evecs = []

    for v in norm_evecs:
        real_v = np.real_if_close(v)
        if (real_v < 0).all():
            real_v *= -1
        real_evecs.append(real_v)

    # skip sorting for now: argsort is pain because numpy will typecase to complex arr
    #    desc_idx = np.argsort(real_evals)[::-1]
    #   return real_evals[desc_idx], real_evecs[desc_idx]
    return real_evals, real_evecs

def calc_spectral_gaps(order, trials=1, n_sample_step=1000):
    """Approximates the spectral gap for each sampler at a certain order
    returns avg_discrete_sg, discrete_sg_var,  avg_continuous_sg, continuous_sg_var
    """
    assert order % 2 == 0
    # normally distributed?
    H = np.random.randn(order / 2)
    c_sg = np.zeros(trials)
    h_sg = np.zeros(trials)

    print "Order: {}".format(order)

    for i in xrange(trials):
        hmc = AlgebraicDiscrete(order, energies=H)
        chmc = AlgebraicContinuous(order, energies=H)
        # runs until close to equilibrium distribution
        n_hmc = hmc.calculate_mixing_time()
        n_chmc = chmc.calculate_mixing_time()
        h_sg[i] = sg(hmc)
        c_sg[i] = sg(chmc)
        print "{} samplings steps for hmc to approach equilibirium".format(n_hmc)
        print "{} samplings steps for chmc to approach equilibirium".format(n_chmc)

    return np.mean(h_sg), np.std(h_sg), np.mean(c_sg), np.std(c_sg)

def sg(sampler):
    """returns the spectral gap
    t: transition matrix
    """
    while True:
        try:
            t = sampler.get_empirical_transition_matrix()
            w,v = eig(t)
            w_ord = np.sort(w)[::-1]
            if np.around(np.real_if_close(w_ord[0]), decimals=5) != 1:
                raise Exception("no eval with value 1")
            return 1 - np.absolute(w_ord[1])
        except RuntimeError:
            sampler.sample(1000)

def plot_sgs(max_ord=100):
    """Saves a plot of spectral gap against order
    """
    plt.clf()
    plt.ion()
    orders = np.arange(2, max_ord) * 2
    sgs = [calc_spectral_gaps(o) for o in orders]
    avg_h_sg, std_h_sg, avg_c_sg, std_c_sg = zip(*sgs)
    plt.errorbar(orders, avg_h_sg, yerr=std_h_sg, label='Discrete sampler')
    plt.errorbar(orders, avg_c_sg, yerr=std_c_sg, label='Continuous sampler')
    plt.title("Spectral gaps on random gaussian state ladders")
    plt.legend()
