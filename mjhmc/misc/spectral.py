"""
Utilities relating to eigenvalue spectrum
Experimental platform
"""

import numpy as np
from mjhmc.samplers.algebraic_hmc import AlgebraicDiscrete

def check_hermitian(operator):
    """ Checks that operator is hermitian (equal to its complex conjugate)

    :param operator: an n x n array
    :returns: true if operator is hermitian
    :rtype: boolean

    """
    return (operator == np.conjugate(operator).T).all()

def mc_herm_search(n_dims, n_draws=10000):
    """ tests energies drawn uniformly at random to see if the corresponding markov transition

    :param n_dims: dimension of space
    :param n_draws: number of energies to draw
    :returns: list of energies that have a hermitian markov matrix
    :rtype: [np.array, ... , np.array]

    """
    herm_energies = {}
    for _ in xrange(n_draws):
        energies = np.random.randn(n_dims)
        fsampler = AlgebraicDiscrete(n_dims * 2, energies)
        if check_hermitian(fsampler.calculate_true_transition_matrix()):
            herm_energies[str(energies)] = energies
    return herm_energies.values()
